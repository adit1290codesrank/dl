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
    float stddev = sqrtf(2.0f / dimension);
    std::normal_distribution<float> dist(0.0f, stddev);

    wQ = Tensor::random({dimension, dimension}, -0.02f, 0.02f);
    wK = Tensor::random({dimension, dimension}, -0.02f, 0.02f);
    wV = Tensor::random({dimension, dimension}, -0.02f, 0.02f);
    wO = Tensor::random({dimension, dimension}, -0.02f, 0.02f);
    
    w_gate = Tensor::random({dimension, heads}, -0.02f, 0.02f);
    w_proj = Tensor::random({2048, dimension}, -0.02f, 0.02f);

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

    Tensor Q = Tensor({N*T_q, D}); matrix_multiply_cuda(q_2d.data(), false, wQ.data(), false, Q.data(), N*T_q, D, D);
    Tensor K = Tensor({N*T_k, D}); matrix_multiply_cuda(k_2d.data(), false, wK.data(), false, K.data(), N*T_k, D, D);
    Tensor V = Tensor({N*T_k, D}); matrix_multiply_cuda(v_2d.data(), false, wV.data(), false, V.data(), N*T_k, D, D);

    cachedQ = Q; cachedK = K; cachedV = V;
    
    Tensor K_total = Tensor({N*T_k, D});
    if (K_frozen.total_elements() > 0) {
        Tensor gate = Tensor({N*T_k, heads});
        matrix_multiply_cuda(k_2d.data(), false, w_gate.data(), false, gate.data(), N*T_k, heads, D);
        sigmoid_inplace_cuda(gate.data(), N*T_k*heads);
        cached_gate = gate;
        
        Tensor kfp = Tensor({T_k, D});
        matrix_multiply_cuda(K_frozen.data(), false, w_proj.data(), false, kfp.data(), T_k, D, 2048);
        cached_K_frozen_proj = kfp;
        
        cached_K_learned = K;
        
        add_gated_k_frozen_cuda(K_total.data(), K.data(), kfp.data(), gate.data(), N, T_k, heads, head_dim);
    } else {
        cudaMemcpy(K_total.data(), K.data(), K.total_elements() * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    Tensor Qh({N*heads, T_q, head_dim});
    Tensor Kh({N*heads, T_k, head_dim});
    Tensor Vh({N*heads, T_k, head_dim});
    
    transpose_multihead_cuda(Q.data(), Qh.data(), N, T_q, heads, head_dim);
    transpose_multihead_cuda(K_total.data(), Kh.data(), N, T_k, heads, head_dim);
    transpose_multihead_cuda(V.data(), Vh.data(), N, T_k, heads, head_dim);

    Tensor scores({N*heads, T_q, T_k});
    batched_matmul_cuda(Qh.data(), false, Kh.data(), true, scores.data(), N*heads, T_q, head_dim, T_k);
    attention_scale_cuda(scores.data(), scale, N*heads*T_q*T_k);

    Tensor attn({N*heads, T_q, T_k});
    cudaMemcpy(attn.data(), scores.data(), N*heads*T_q*T_k * sizeof(float), cudaMemcpyDeviceToDevice);
    full_attention_softmax_cuda(attn.data(), N*heads*T_q, T_k);
    cached_attention = attn;

    Tensor context({N*heads, T_q, head_dim});
    batched_matmul_cuda(attn.data(), false, Vh.data(), false, context.data(), N*heads, T_q, T_k, head_dim);

    Tensor merged(N*T_q, D);
    untranspose_multihead_cuda(context.data(), merged.data(), N, T_q, heads, head_dim);

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

    Tensor dContext({N*heads, T_q, head_dim});
    transpose_multihead_cuda(dMerged.data(), dContext.data(), N, T_q, heads, head_dim);

    Tensor dAttn({N*heads, T_q, T_k});
    batched_matmul_cuda(dContext.data(), false, cachedV.data(), true, dAttn.data(), N*heads, T_q, head_dim, T_k);

    Tensor dVh({N*heads, T_k, head_dim});
    batched_matmul_cuda(cached_attention.data(), true, dContext.data(), false, dVh.data(), N*heads, T_k, T_q, head_dim);

    Tensor dScores({N*heads, T_q, T_k});
    full_attention_softmax_backward_cuda(dAttn.data(), cached_attention.data(), dScores.data(), N*heads*T_q, T_k);

    attention_scale_cuda(dScores.data(), scale, N*heads*T_q*T_k);

    Tensor dQh({N*heads, T_q, head_dim});
    Tensor dKh({N*heads, T_k, head_dim});
    batched_matmul_cuda(dScores.data(), false, cachedK.data(), false, dQh.data(), N*heads, T_q, T_k, head_dim);
    batched_matmul_cuda(dScores.data(), true, cachedQ.data(), false, dKh.data(), N*heads, T_k, T_q, head_dim);

    Tensor dQ({N*T_q, D});
    Tensor dK_total({N*T_k, D});
    untranspose_multihead_cuda(dQh.data(), dQ.data(), N, T_q, heads, head_dim);
    untranspose_multihead_cuda(dKh.data(), dK_total.data(), N, T_k, heads, head_dim);

    Tensor dK = Tensor::zeros(N*T_k, D);
    cudaMemset(dw_gate.data(), 0, dw_gate.total_elements() * sizeof(float));
    cudaMemset(dw_proj.data(), 0, dw_proj.total_elements() * sizeof(float));
    
    if (K_frozen.total_elements() > 0) {
        Tensor dGate = Tensor::zeros(N*T_k, heads);
        Tensor dK_frozen_proj = Tensor::zeros(T_k, D);
        
        backward_gated_k_frozen_cuda(dK_total.data(), cached_K_frozen_proj.data(), cached_gate.data(),
                                     dK.data(), dGate.data(), dK_frozen_proj.data(),
                                     N, T_k, heads, head_dim);
                                     
        matrix_multiply_cuda(K_frozen.data(), true, dK_frozen_proj.data(), false, dw_proj.data(), 2048, T_k, D);
        
        sigmoid_backward_inplace_cuda(dGate.data(), cached_gate.data(), N*T_k*heads);
        
        Tensor schema_2d = cached_schema; schema_2d.shape = {N*T_k, D};
        matrix_multiply_cuda(schema_2d.data(), true, dGate.data(), false, dw_gate.data(), D, N*T_k, heads);
    } else {
        cudaMemcpy(dK.data(), dK_total.data(), dK_total.total_elements() * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    Tensor q_2d = cached_query.reshape({N*T_q, D});
    Tensor k_2d = cached_schema.reshape({N*T_k, D});

    Tensor dV({N*T_k, D});
    untranspose_multihead_cuda(dVh.data(), dV.data(), N, T_k, heads, head_dim);

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
        // We will just do a quick CPU add for this small vector
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
