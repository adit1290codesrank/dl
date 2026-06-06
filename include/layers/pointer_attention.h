#pragma once
#include "layer.h"
#include <fstream>
#include "../core/tensor.h"

// PointerAttention implements the Cross-Attention Schema-Linking RAG Layer
// It calculates Similarity = Softmax(Q * K^T) and returns the attention weights
// so the decoder can use them as Pointer probabilities.
class PointerAttention : public Layer
{
    private:
        int dimension;
        int heads;
        float scale;

        Tensor wQ, wK, wV, wO;
        Tensor dwQ, dwK, dwV, dwO;
        
        Tensor w_gate, dw_gate; // Shape [dimension, heads]
        Tensor w_proj, dw_proj; // Shape [2048, dimension]

        int t;
        Tensor mwq, vwq;
        Tensor mwk, vwk;
        Tensor mwv, vwv;
        Tensor mwo, vwo;
        Tensor mw_gate, vw_gate;
        Tensor mw_proj, vw_proj;

        Tensor cached_query;
        Tensor cached_schema;
        Tensor cached_gate; // [N, T_k, heads]
        Tensor cached_K_learned; // [N, T_k, dimension]
        Tensor cached_K_frozen_proj; // [T_k, dimension]
        Tensor cachedQ, cachedK, cachedV;
        Tensor cached_attention;
        
        Tensor K_frozen; // Global frozen vectors [schema_size, 2048]
        Tensor cached_dSchema; // Gradient for schema encoder input

    public:
        // dimension is the embedding dim.
        PointerAttention(int dimension, int heads);
        
        // standard forward is overridden but we'll use a specific dual-forward
        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad, float lr) override;
        Tensor backward_ext(const Tensor& grad, float lr, const Tensor* d_attn_ext);
        Tensor get_schema_grad() const { return cached_dSchema; }
        void set_k_frozen(const Tensor& kf) { K_frozen = kf; }

        // Dual forward taking Query and Schema
        // returns {context, attention_weights}
        std::pair<Tensor, Tensor> forward_dual(const Tensor& query, const Tensor& schema);
        
        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
};

// Custom CUDA kernel launcher for scatter-adding the pointer probabilities
// P_final = p_gen * P_vocab + (1 - p_gen) * P_schema
void pointer_scatter_add_cuda(
    const float* p_vocab, 
    const float* p_schema, 
    const int* schema_vocab_indices,
    float* p_final,
    float p_gen,
    int vocab_size,
    int schema_size,
    int batch_size
);
