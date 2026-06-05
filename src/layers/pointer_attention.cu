#include "../../include/layers/pointer_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

// External CUDA kernel declarations used from the framework
Tensor matrix_multiply(const Tensor& A, bool transA, const Tensor& B, bool transB);
void adam_cuda(Tensor& W, const Tensor& grad, Tensor& m, Tensor& v, float lr, int t, int size, float lamda=0.001f);
void attention_scale_cuda(float* data, float scale, int size);
void full_attention_softmax_cuda(float* data, int rows, int cols);
void full_attention_softmax_backward_cuda(const float* dY, const float* Y, float* dX, int rows, int cols);
void split_heads_cuda(const float* input, float* output, int N, int T, int H, int dk);
void merge_heads_cuda(const float* input, float* output, int N, int T, int H, int dk);
void batched_matmul_cuda(const float* A, bool transA, const float* B, bool transB, float* C, int batch, int M, int K, int N);

// -------------------------------------------------------------------------
// Gated K_frozen (Lexical Keys) Kernels
// -------------------------------------------------------------------------
__global__ void add_gated_k_frozen_kernel(float* K_total, const float* K_learned, const float* K_frozen_proj, const float* gate, int N, int T_k, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dimension = heads * head_dim;
    int total = N * T_k * dimension;
    if (idx < total) {
        int d = idx % dimension;
        int t = (idx / dimension) % T_k;
        int n = idx / (dimension * T_k);
        int h = d / head_dim;
        
        float g = gate[(n * T_k + t) * heads + h];
        float kf = K_frozen_proj[t * dimension + d];
        float kl = K_learned[idx];
        
        K_total[idx] = kl + g * kf;
    }
}

void add_gated_k_frozen_cuda(float* K_total, const float* K_learned, const float* K_frozen_proj, const float* gate, int N, int T_k, int heads, int head_dim) {
    int total = N * T_k * heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_gated_k_frozen_kernel<<<blocks, threads>>>(K_total, K_learned, K_frozen_proj, gate, N, T_k, heads, head_dim);
    cudaDeviceSynchronize();
}

__global__ void backward_gated_k_frozen_kernel(const float* dK_total, const float* K_frozen_proj, const float* gate, 
                                              float* dK_learned, float* dGate, float* dK_frozen_proj,
                                              int N, int T_k, int heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dimension = heads * head_dim;
    int total = N * T_k * dimension;
    if (idx < total) {
        int d = idx % dimension;
        int t = (idx / dimension) % T_k;
        int n = idx / (dimension * T_k);
        int h = d / head_dim;
        
        float dkt = dK_total[idx];
        dK_learned[idx] = dkt;
        
        int gate_idx = (n * T_k + t) * heads + h;
        float kf = K_frozen_proj[t * dimension + d];
        atomicAdd(&dGate[gate_idx], dkt * kf);
        
        int kfp_idx = t * dimension + d;
        float g = gate[gate_idx];
        atomicAdd(&dK_frozen_proj[kfp_idx], dkt * g);
    }
}

void backward_gated_k_frozen_cuda(const float* dK_total, const float* K_frozen_proj, const float* gate, 
                                 float* dK_learned, float* dGate, float* dK_frozen_proj,
                                 int N, int T_k, int heads, int head_dim) {
    int total = N * T_k * heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    backward_gated_k_frozen_kernel<<<blocks, threads>>>(dK_total, K_frozen_proj, gate, dK_learned, dGate, dK_frozen_proj, N, T_k, heads, head_dim);
    cudaDeviceSynchronize();
}

__global__ void sigmoid_inplace_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void sigmoid_backward_inplace_kernel(float* dx, const float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        dx[idx] = dx[idx] * val * (1.0f - val);
    }
}

void sigmoid_inplace_cuda(float* x, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_inplace_kernel<<<blocks, threads>>>(x, size);
    cudaDeviceSynchronize();
}

void sigmoid_backward_inplace_cuda(float* dx, const float* x, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sigmoid_backward_inplace_kernel<<<blocks, threads>>>(dx, x, size);
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------------
// Custom CUDA Kernels for Pointer-Generator
// -------------------------------------------------------------------------

__global__ void pointer_base_dist_kernel(const float* p_vocab, float* p_final, float p_gen, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        p_final[idx] = p_gen * p_vocab[idx];
    }
}

__global__ void pointer_scatter_add_kernel(const float* p_schema, const int* schema_indices, float* p_final, float p_gen, int schema_size, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < schema_size) {
        int target_vocab_idx = schema_indices[idx];
        if(target_vocab_idx >= 0 && target_vocab_idx < vocab_size) {
            float val = (1.0f - p_gen) * p_schema[idx];
            // Hardware-accelerated atomic add (efficient on sm_89 L2 cache)
            atomicAdd(&p_final[target_vocab_idx], val);
        }
    }
}

void pointer_scatter_add_cuda(const float* p_vocab, const float* p_schema, const int* schema_vocab_indices, float* p_final, float p_gen, int vocab_size, int schema_size, int batch_size) {
    // Note: Assuming batch_size = 1 for simplicity in inference
    int threads = 256;
    int blocks_vocab = (vocab_size + threads - 1) / threads;
    pointer_base_dist_kernel<<<blocks_vocab, threads>>>(p_vocab, p_final, p_gen, vocab_size);
    cudaDeviceSynchronize();

    if(schema_size > 0) {
        int blocks_schema = (schema_size + threads - 1) / threads;
        pointer_scatter_add_kernel<<<blocks_schema, threads>>>(p_schema, schema_vocab_indices, p_final, p_gen, schema_size, vocab_size);
        cudaDeviceSynchronize();
    }
}

// -------------------------------------------------------------------------
// PointerAttention Class Implementation
// -------------------------------------------------------------------------

PointerAttention::PointerAttention(int dimension, int heads) : dimension(dimension), heads(heads), t(0)
{
    if(dimension % heads != 0) throw std::invalid_argument("Dimension must be divisible by heads");
    int head_dim = dimension / heads;
    scale = 1.0f / sqrtf((float)head_dim);

    std::mt19937 gen(42);
    
    // Standard He initialization for attention projections (fan_in = dimension)
    float stddev_attn = sqrtf(2.0f / dimension);
    std::normal_distribution<float> dist_attn(0.0f, stddev_attn);
    auto init_attn = [&](int r, int c) -> Tensor {
        std::vector<float> h(r*c);
        for(int i=0; i<r*c; i++) h[i] = dist_attn(gen);
        return Tensor::upload(h, r, c);
    };

    wQ = init_attn(dimension, dimension);
    wK = init_attn(dimension, dimension);
    wV = init_attn(dimension, dimension);
    wO = init_attn(dimension, dimension);
    
    // w_gate feeds into Sigmoid: Use Xavier/Glorot initialization
    float stddev_gate = sqrtf(2.0f / (dimension + heads));
    std::normal_distribution<float> dist_gate(0.0f, stddev_gate);
    std::vector<float> h_gate(dimension * heads);
    for(int i=0; i<dimension*heads; i++) h_gate[i] = dist_gate(gen);
    w_gate = Tensor::upload(h_gate, {dimension, heads});

    // w_proj is linear from 2048: Use He initialization with fan_in = 2048
    float stddev_proj = sqrtf(2.0f / 2048.0f);
    std::normal_distribution<float> dist_proj(0.0f, stddev_proj);
    std::vector<float> h_proj(2048 * dimension);
    for(int i=0; i<2048*dimension; i++) h_proj[i] = dist_proj(gen);
    w_proj = Tensor::upload(h_proj, {2048, dimension});

    dwQ = Tensor::zeros(dimension, dimension);
    dwK = Tensor::zeros(dimension, dimension);
    dwV = Tensor::zeros(dimension, dimension);
    dwO = Tensor::zeros(dimension, dimension);
    dw_gate = Tensor::zeros(dimension, heads);
    dw_proj = Tensor::zeros(2048, dimension);

    mwq = Tensor::zeros(dimension, dimension); vwq = Tensor::zeros(dimension, dimension);
    mwk = Tensor::zeros(dimension, dimension); vwk = Tensor::zeros(dimension, dimension);
    mwv = Tensor::zeros(dimension, dimension); vwv = Tensor::zeros(dimension, dimension);
    mwo = Tensor::zeros(dimension, dimension); vwo = Tensor::zeros(dimension, dimension);
    mw_gate = Tensor::zeros(dimension, heads); vw_gate = Tensor::zeros(dimension, heads);
    mw_proj = Tensor::zeros(2048, dimension); vw_proj = Tensor::zeros(2048, dimension);
}

Tensor PointerAttention::forward(const Tensor& input) {
    int D = input.shape.back();
    int N = (input.shape.size() == 3) ? input.shape[0] : 1;
    int T = (input.shape.size() == 3) ? input.shape[1] : (int)(input.total_elements()/D);
    return forward_dual(input, input).first;
}

std::pair<Tensor, Tensor> PointerAttention::forward_dual(const Tensor& query, const Tensor& schema) {
    cached_query = query;
    cached_schema = schema;

    int D = query.shape.back();
    int N = (query.shape.size() == 3) ? query.shape[0] : 1;
    int T_q = (query.shape.size() == 3) ? query.shape[1] : (int)(query.total_elements()/D);
    int T_k = (schema.shape.size() == 3) ? schema.shape[1] : (int)(schema.total_elements()/D);
    int head_dim = D / heads;

    Tensor q_2d = query; q_2d.shape = {N*T_q, D};
    Tensor k_2d = schema; k_2d.shape = {N*T_k, D};
    Tensor v_2d = schema; v_2d.shape = {N*T_k, D};

    Tensor Q = matrix_multiply(q_2d, false, wQ, false);
    Tensor K = matrix_multiply(k_2d, false, wK, false);
    Tensor V = matrix_multiply(v_2d, false, wV, false);

    cachedQ = Q; cachedK = K; cachedV = V;
    
    Tensor K_total = Tensor::zeros(N*T_k, D);
    if (K_frozen.total_elements() > 0) {
        Tensor gate = matrix_multiply(k_2d, false, w_gate, false);
        sigmoid_inplace_cuda(gate.data(), N*T_k*heads);
        cached_gate = gate;
        
        Tensor kfp = matrix_multiply(K_frozen, false, w_proj, false);
        cached_K_frozen_proj = kfp;
        
        cached_K_learned = K;
        
        add_gated_k_frozen_cuda(K_total.data(), K.data(), kfp.data(), gate.data(), N, T_k, heads, head_dim);
    } else {
        cudaMemcpy(K_total.data(), K.data(), K.total_elements() * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    Tensor Qh = Tensor({N*heads, T_q, head_dim});
    Tensor Kh = Tensor({N*heads, T_k, head_dim});
    Tensor Vh = Tensor({N*heads, T_k, head_dim});
    
    split_heads_cuda(Q.data(), Qh.data(), N, T_q, heads, head_dim);
    split_heads_cuda(K_total.data(), Kh.data(), N, T_k, heads, head_dim);
    split_heads_cuda(V.data(), Vh.data(), N, T_k, heads, head_dim);

    Tensor scores = Tensor({N*heads, T_q, T_k});
    batched_matmul_cuda(Qh.data(), false, Kh.data(), true, scores.data(), N*heads, T_q, head_dim, T_k);
    attention_scale_cuda(scores.data(), scale, N*heads*T_q*T_k);

    Tensor attn = Tensor({N*heads, T_q, T_k});
    cudaMemcpy(attn.data(), scores.data(), N*heads*T_q*T_k * sizeof(float), cudaMemcpyDeviceToDevice);
    full_attention_softmax_cuda(attn.data(), N*heads*T_q, T_k);
    cached_attention = attn;

    Tensor context = Tensor({N*heads, T_q, head_dim});
    batched_matmul_cuda(attn.data(), false, Vh.data(), false, context.data(), N*heads, T_q, T_k, head_dim);

    Tensor merged(N*T_q, D);
    merge_heads_cuda(context.data(), merged.data(), N, T_q, heads, head_dim);

    Tensor output = matrix_multiply(merged, false, wO, false);
    if(query.shape.size() == 3) output.shape = {N, T_q, D};
    
    return {output, attn};
}

Tensor PointerAttention::backward(const Tensor& grad, float lr) {
    t++;
    int D = cached_query.shape.back();
    int N = (cached_query.shape.size() == 3) ? cached_query.shape[0] : 1;
    int T_q = (cached_query.shape.size() == 3) ? cached_query.shape[1] : (int)(cached_query.total_elements()/D);
    int T_k = (cached_schema.shape.size() == 3) ? cached_schema.shape[1] : (int)(cached_schema.total_elements()/D);
    int head_dim = D / heads;

    Tensor dY_flat = grad.reshape({N*T_q, D});
    Tensor dMerged = matrix_multiply(dY_flat, false, wO, true);
    dwO = matrix_multiply(cached_query.reshape({N*T_q, D}), true, dY_flat, false);

    Tensor dContext = Tensor({N*heads, T_q, head_dim});
    split_heads_cuda(dMerged.data(), dContext.data(), N, T_q, heads, head_dim);

    Tensor dAttn = Tensor({N*heads, T_q, T_k});
    batched_matmul_cuda(dContext.data(), false, cachedV.data(), true, dAttn.data(), N*heads, T_q, head_dim, T_k);

    Tensor dVh = Tensor({N*heads, T_k, head_dim});
    batched_matmul_cuda(cached_attention.data(), true, dContext.data(), false, dVh.data(), N*heads, T_k, T_q, head_dim);

    Tensor dScores = Tensor({N*heads, T_q, T_k});
    full_attention_softmax_backward_cuda(dAttn.data(), cached_attention.data(), dScores.data(), N*heads*T_q, T_k);

    attention_scale_cuda(dScores.data(), scale, N*heads*T_q*T_k);

    Tensor dQh = Tensor({N*heads, T_q, head_dim});
    Tensor dKh = Tensor({N*heads, T_k, head_dim});
    batched_matmul_cuda(dScores.data(), false, cachedK.data(), false, dQh.data(), N*heads, T_q, T_k, head_dim);
    batched_matmul_cuda(dScores.data(), true, cachedQ.data(), false, dKh.data(), N*heads, T_k, T_q, head_dim);

    Tensor dQ = Tensor({N*T_q, D});
    Tensor dK_total = Tensor({N*T_k, D});
    merge_heads_cuda(dQh.data(), dQ.data(), N, T_q, heads, head_dim);
    merge_heads_cuda(dKh.data(), dK_total.data(), N, T_k, heads, head_dim);

    Tensor dK = Tensor::zeros(N*T_k, D);
    cudaMemset(dw_gate.data(), 0, dw_gate.total_elements() * sizeof(float));
    cudaMemset(dw_proj.data(), 0, dw_proj.total_elements() * sizeof(float));
    
    if (K_frozen.total_elements() > 0) {
        Tensor dGate = Tensor::zeros(N*T_k, heads);
        Tensor dK_frozen_proj = Tensor::zeros(T_k, D);
        
        backward_gated_k_frozen_cuda(dK_total.data(), cached_K_frozen_proj.data(), cached_gate.data(),
                                     dK.data(), dGate.data(), dK_frozen_proj.data(),
                                     N, T_k, heads, head_dim);
        
        Tensor dw_proj_new = matrix_multiply(K_frozen, true, dK_frozen_proj, false);
        cudaMemcpy(dw_proj.data(), dw_proj_new.data(), dw_proj.total_elements() * sizeof(float), cudaMemcpyDeviceToDevice);
        
        sigmoid_backward_inplace_cuda(dGate.data(), cached_gate.data(), N*T_k*heads);
        
        Tensor schema_2d = cached_schema; schema_2d.shape = {N*T_k, D};
        
        // dw_gate = schema_emb^T @ dGate
        // Using matrix_multiply wrappers to get allocated result then memcpy to dw_gate
        Tensor dw_gate_new = matrix_multiply(schema_2d, true, dGate, false);
        cudaMemcpy(dw_gate.data(), dw_gate_new.data(), dw_gate.total_elements() * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // dSchema_gate = dGate @ w_gate^T
        Tensor dSchema_gate = matrix_multiply(dGate, false, w_gate, true);
        cached_gate = dSchema_gate; // Reuse tensor memory for passing to dInput_k
    } else {
        cudaMemcpy(dK.data(), dK_total.data(), dK_total.total_elements() * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    Tensor q_2d = cached_query.reshape({N*T_q, D});
    Tensor k_2d = cached_schema.reshape({N*T_k, D});

    Tensor dV = Tensor({N*T_k, D});
    merge_heads_cuda(dVh.data(), dV.data(), N, T_k, heads, head_dim);

    dwQ = matrix_multiply(q_2d, true, dQ, false);
    dwK = matrix_multiply(k_2d, true, dK, false);
    dwV = matrix_multiply(k_2d, true, dV, false);
    
    Tensor dInput_q = matrix_multiply(dQ, false, wQ, true);
    
    // dInput_k = dK @ wK^T
    // dInput_v = dV @ wV^T
    Tensor dInput_k = matrix_multiply(dK, false, wK, true);
    Tensor dInput_v = matrix_multiply(dV, false, wV, true);
    
    if (K_frozen.total_elements() > 0) {
        // Add dSchema_gate (currently cached_gate) to dInput_k
        std::vector<float> h_in = dInput_k.download();
        std::vector<float> h_gate = cached_gate.download();
        for(size_t i=0; i<h_in.size(); ++i) h_in[i] += h_gate[i];
        cudaMemcpy(dInput_k.data(), h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    // 1. Adam Update
    int wsize = dimension * dimension;
    adam_cuda(wQ, dwQ, mwq, vwq, lr, t, wsize);
    adam_cuda(wK, dwK, mwk, vwk, lr, t, wsize);
    adam_cuda(wV, dwV, mwv, vwv, lr, t, wsize);
    adam_cuda(wO, dwO, mwo, vwo, lr, t, wsize);
    adam_cuda(w_gate, dw_gate, mw_gate, vw_gate, lr, t, dimension * heads);
    adam_cuda(w_proj, dw_proj, mw_proj, vw_proj, lr, t, 2048 * dimension);

    if (cached_query.shape.size() == 3) {
        dInput_q.shape = {N, T_q, D};
    }
    
    return dInput_q;
}

void PointerAttention::save(std::ofstream& os) {
    auto s=[&](const Tensor& t){std::vector<float> h=t.download();os.write(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(float));};
    s(wQ); s(wK); s(wV); s(wO); s(w_gate); s(w_proj);
}

void PointerAttention::load(std::ifstream& is) {
    auto l=[&](Tensor& t){std::vector<float> h=t.download();is.read(reinterpret_cast<char*>(h.data()),h.size()*sizeof(float));t=Tensor::upload(h, t.shape);};
    l(wQ); l(wK); l(wV); l(wO);
    l(w_gate); l(w_proj);
}
