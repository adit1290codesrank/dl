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

    auto init_weight=[&](int r,int c) -> Tensor {
        std::vector<float> h(r*c);
        for(int i=0; i<r*c; i++) h[i] = dist(gen);
        return Tensor::upload(h, r, c);
    };

    wQ = init_weight(dimension, dimension);
    wK = init_weight(dimension, dimension);
    wV = init_weight(dimension, dimension);
    wO = init_weight(dimension, dimension);

    dwQ = Tensor::zeros(dimension, dimension);
    dwK = Tensor::zeros(dimension, dimension);
    dwV = Tensor::zeros(dimension, dimension);
    dwO = Tensor::zeros(dimension, dimension);

    mwq = Tensor::zeros(dimension, dimension); vwq = Tensor::zeros(dimension, dimension);
    mwk = Tensor::zeros(dimension, dimension); vwk = Tensor::zeros(dimension, dimension);
    mwv = Tensor::zeros(dimension, dimension); vwv = Tensor::zeros(dimension, dimension);
    mwo = Tensor::zeros(dimension, dimension); vwo = Tensor::zeros(dimension, dimension);
}

Tensor PointerAttention::forward(const Tensor& input) {
    // Should use forward_dual. If called directly, treat as self-attention
    return forward_dual(input, input).first;
}

std::pair<Tensor, Tensor> PointerAttention::forward_dual(const Tensor& query, const Tensor& schema) {
    cached_query = query;
    cached_schema = schema;

    int D = query.shape.back();
    int N = (query.shape.size() == 3) ? query.shape[0] : 1;
    int T_q = (query.shape.size() == 3) ? query.shape[1] : (int)(query.total_elements()/D);
    int T_k = (schema.shape.size() == 3) ? schema.shape[1] : (int)(schema.total_elements()/D);
    
    int H = heads;
    int head_dim = D / H;

    Tensor q_2d = query.reshape({N*T_q, D});
    Tensor k_2d = schema.reshape({N*T_k, D});

    Tensor Q = matrix_multiply(q_2d, false, wQ, false);
    Tensor K = matrix_multiply(k_2d, false, wK, false);
    Tensor V = matrix_multiply(k_2d, false, wV, false);

    Tensor Qh({N*H, T_q, head_dim});
    Tensor Kh({N*H, T_k, head_dim});
    Tensor Vh({N*H, T_k, head_dim});
    
    split_heads_cuda(Q.data(), Qh.data(), N, T_q, H, head_dim);
    split_heads_cuda(K.data(), Kh.data(), N, T_k, H, head_dim);
    split_heads_cuda(V.data(), Vh.data(), N, T_k, H, head_dim);

    cachedQ = Qh;
    cachedK = Kh;
    cachedV = Vh;

    // Scores = Q @ K^T -> (N*H, T_q, T_k)
    Tensor scores({N*H, T_q, T_k});
    batched_matmul_cuda(Qh.data(), false, Kh.data(), true, scores.data(), N*H, T_q, head_dim, T_k);
    attention_scale_cuda(scores.data(), scale, N*H*T_q*T_k);

    // Softmax
    Tensor attn({N*H, T_q, T_k});
    cudaMemcpy(attn.data(), scores.data(), N*H*T_q*T_k * sizeof(float), cudaMemcpyDeviceToDevice);
    full_attention_softmax_cuda(attn.data(), N*H*T_q, T_k);
    cached_attention = attn;

    // Context = Attention @ V -> (N*H, T_q, head_dim)
    Tensor context({N*H, T_q, head_dim});
    batched_matmul_cuda(attn.data(), false, Vh.data(), false, context.data(), N*H, T_q, T_k, head_dim);

    // Merge heads
    Tensor merged(N*T_q, D);
    merge_heads_cuda(context.data(), merged.data(), N, T_q, H, head_dim);

    Tensor output = matrix_multiply(merged, false, wO, false);
    if(query.shape.size() == 3) output.shape = {N, T_q, D};
    
    return {output, attn};
}

Tensor PointerAttention::backward(const Tensor& grad, float lr) {
    // Skipping full backward pass logic for brevity in planning
    // A complete implementation would calculate dQ, dK, dV.
    return grad;
}

void PointerAttention::save(std::ofstream& os) {
    auto s=[&](const Tensor& t){std::vector<float> h=t.download();os.write(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(float));};
    s(wQ); s(wK); s(wV); s(wO);
}

void PointerAttention::load(std::ifstream& is) {
    auto l=[&](Tensor& t){std::vector<float> h(dimension*dimension);is.read(reinterpret_cast<char*>(h.data()),h.size()*sizeof(float));t=Tensor::upload(h,dimension,dimension);};
    l(wQ); l(wK); l(wV); l(wO);
}
