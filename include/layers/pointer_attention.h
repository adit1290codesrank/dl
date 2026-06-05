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

        int t;
        Tensor mwq, vwq;
        Tensor mwk, vwk;
        Tensor mwv, vwv;
        Tensor mwo, vwo;

        Tensor cached_query;
        Tensor cached_schema;
        Tensor cachedQ, cachedK, cachedV;
        Tensor cached_attention;

    public:
        // dimension is the embedding dim.
        PointerAttention(int dimension, int heads);
        
        // standard forward is overridden but we'll use a specific dual-forward
        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad, float lr) override;

        // Dual forward taking Query (English) and Schema (Macros/Columns)
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
