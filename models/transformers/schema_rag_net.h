#pragma once
#include "text_encoder.h"
#include "../../include/layers/pointer_attention.h"
#include <iostream>

// SchemaRAGNet implements the Dual-Encoder Architecture with Pointer-Generator Decoder
class SchemaRAGNet
{
    private:
        TextEncoder* query_encoder;
        TextEncoder* schema_encoder;
        PointerAttention* pointer_layer;

        int vocab_size;
        int max_seq_len;
        int dimension;
        int heads;
        int depth;

    public:
        SchemaRAGNet(int vocab_size, int max_seq_len, int dimension, int heads, int depth)
            : vocab_size(vocab_size), max_seq_len(max_seq_len), dimension(dimension), heads(heads), depth(depth)
        {
            // The dual encoders (query and schema share same architecture but different weights)
            query_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth);
            schema_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth);
            
            // Explicit Schema-RAG Pointer Layer
            pointer_layer = new PointerAttention(dimension, heads);
        }

        ~SchemaRAGNet()
        {
            delete query_encoder;
            delete schema_encoder;
            delete pointer_layer;
        }

        // Forward pass takes tokenized English Query and tokenized Schema
        // Returns the final merged logits for the extended vocabulary
        Tensor forward(const Tensor& query_tokens, const Tensor& schema_tokens)
        {
            // 1. Dual Encode
            Tensor Q_emb = query_encoder->forward(query_tokens);
            Tensor K_emb = schema_encoder->forward(schema_tokens);

            // 2. Schema-RAG Layer (Pointer Attention)
            // returns pairs of {Context, Attention_Weights}
            auto result = pointer_layer->forward_dual(Q_emb, K_emb);
            Tensor context = result.first;
            Tensor attention_weights = result.second; // This is our P_schema

            // NOTE: In a full implementation, the context would be passed through a feed-forward layer 
            // to generate P_vocab, and a linear layer to generate p_gen. 
            // Then `pointer_scatter_add_cuda` would be called to merge P_vocab and P_schema.
            
            return context; 
        }

        void save(const std::string& path)
        {
            query_encoder->save(path + "_query.bin");
            schema_encoder->save(path + "_schema.bin");
            std::ofstream os(path + "_pointer.bin", std::ios::binary);
            pointer_layer->save(os);
            os.close();
        }

        void load(const std::string& path)
        {
            query_encoder->load(path + "_query.bin");
            schema_encoder->load(path + "_schema.bin");
            std::ifstream is(path + "_pointer.bin", std::ios::binary);
            pointer_layer->load(is);
            is.close();
        }
};
