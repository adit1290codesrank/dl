#pragma once
#include "text_encoder.h"
#include "../../include/layers/pointer_attention.h"
#include "../../include/layers/dense.h"
#include "../../include/layers/softmax.h"
#include "../../include/core/loss.h"
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

// SchemaRAGNet implements the Dual-Encoder Architecture with Pointer-Generator Decoder
class SchemaRAGNet
{
    private:
        TextEncoder* query_encoder;
        TextEncoder* schema_encoder;
        PointerAttention* pointer_layer;
        
        Dense* vocab_proj;
        Softmax* sm;

        int vocab_size;
        int max_seq_len;
        int dimension;
        int heads;
        int depth;

    public:
        SchemaRAGNet(int vocab_size, int max_seq_len, int dimension, int heads, int depth)
            : vocab_size(vocab_size), max_seq_len(max_seq_len), dimension(dimension), heads(heads), depth(depth)
        {
            query_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth);
            schema_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth);
            pointer_layer = new PointerAttention(dimension, heads);
            vocab_proj = new Dense(dimension, vocab_size);
            sm = new Softmax();
        }

        ~SchemaRAGNet()
        {
            delete query_encoder;
            delete schema_encoder;
            delete pointer_layer;
            delete vocab_proj;
            delete sm;
        }

        Tensor forward(const Tensor& query_tokens, const Tensor& schema_tokens)
        {
            Tensor Q_emb = query_encoder->forward(query_tokens);
            Tensor K_emb = schema_encoder->forward(schema_tokens);

            auto result = pointer_layer->forward_dual(Q_emb, K_emb);
            Tensor context = result.first;
            
            // Note: pointer scatter logic omitted for backprop simplicity
            // Standard Seq2Seq Projection
            Tensor logits = vocab_proj->forward(context);
            return sm->forward(logits);
        }

        Tensor backward(const Tensor& dY, float lr)
        {
            Tensor d = sm->backward(dY, lr);
            d = vocab_proj->backward(d, lr);
            d = pointer_layer->backward(d, lr);
            // Propagate only into query_encoder for this simplified sequence pass
            d = query_encoder->backward(d, lr);
            return d;
        }

        void set_mode(bool train)
        {
            query_encoder->set_mode(train);
            schema_encoder->set_mode(train);
            vocab_proj->set_mode(train);
        }

        void save(const std::string& path)
        {
            query_encoder->save(path + "_query.bin");
            schema_encoder->save(path + "_schema.bin");
            std::ofstream os(path + "_pointer.bin", std::ios::binary);
            pointer_layer->save(os);
            os.close();
            std::ofstream os_vocab(path + "_vocab.bin", std::ios::binary);
            vocab_proj->save(os_vocab);
            os_vocab.close();
        }

        void load(const std::string& path)
        {
            query_encoder->load(path + "_query.bin");
            schema_encoder->load(path + "_schema.bin");
            std::ifstream is(path + "_pointer.bin", std::ios::binary);
            pointer_layer->load(is);
            is.close();
            std::ifstream is_vocab(path + "_vocab.bin", std::ios::binary);
            vocab_proj->load(is_vocab);
            is_vocab.close();
        }

        void fit(const std::vector<float>& X, const std::vector<float>& Schema, const std::vector<float>& Y, 
                 int n, int seq_len, int schema_size, int vocab_size, int epochs, int bs, float lr)
        {
            set_mode(true);
            int nb = n / bs;
            if (nb == 0) nb = 1;

            Tensor loss_val = Tensor::zeros(1, 1);
            Tensor grad = Tensor::zeros(bs * seq_len, vocab_size);

            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::mt19937 rng(42);

            std::cout << "Starting SchemaRAG Training Loop..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();

            for (int e = 1; e <= epochs; ++e)
            {
                std::shuffle(idx.begin(), idx.end(), rng);
                float tot_loss = 0.0f;

                for (int b = 0; b < nb; ++b)
                {
                    int current_bs = std::min(bs, n - b * bs);
                    std::vector<float> bX(current_bs * seq_len);
                    std::vector<float> bY(current_bs * seq_len * vocab_size, 0.0f); // One-hot Y
                    std::vector<float> bS(current_bs * schema_size);

                    for (int i = 0; i < current_bs; ++i)
                    {
                        int id = idx[b * bs + i];
                        std::copy(X.begin() + id * seq_len, X.begin() + (id + 1) * seq_len, bX.begin() + i * seq_len);
                        std::copy(Schema.begin() + id * schema_size, Schema.begin() + (id + 1) * schema_size, bS.begin() + i * schema_size);
                        
                        // Create one-hot labels for Sequence output
                        for(int t = 0; t < seq_len; ++t) {
                            int token_id = (int)Y[id * seq_len + t];
                            if(token_id >= 0 && token_id < vocab_size) {
                                bY[(i * seq_len + t) * vocab_size + token_id] = 1.0f;
                            }
                        }
                    }

                    Tensor dX = Tensor::upload(bX, {current_bs, seq_len});
                    Tensor dS = Tensor::upload(bS, {current_bs, schema_size});
                    Tensor dY = Tensor::upload(bY, {current_bs * seq_len, vocab_size});

                    Tensor pred = forward(dX, dS);
                    
                    // Reshape pred from [batch, seq_len, vocab] to [batch*seq_len, vocab] for loss
                    pred.shape = {current_bs * seq_len, vocab_size};
                    
                    Loss::compute_gradient(pred, dY, grad, LossType::CROSS_ENTROPY);
                    backward(grad, lr);
                    tot_loss += Loss::compute_loss(pred, dY, loss_val, LossType::CROSS_ENTROPY);
                    
                    pred.shape = {current_bs, seq_len, vocab_size}; // restore shape

                    if (b % 5 == 0 || b == nb - 1) {
                        int pct = (b * 100) / nb;
                        std::cout << "\rEpoch " << e << "/" << epochs << " [";
                        for (int p = 0; p < 20; p++) {
                            if (p < pct / 5) std::cout << "=";
                            else if (p == pct / 5) std::cout << ">";
                            else std::cout << " ";
                        }
                        std::cout << "] " << pct << "%  Batch " << b << "/" << nb << std::flush;
                    }
                }

                float avg = tot_loss / nb;
                std::cout << "\rEpoch " << e << "/" << epochs << "  Loss: " << avg << "  LR: " << lr << "                            " << std::endl;
            }

            float secs = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "Done in " << secs << "s" << std::endl;
        }
};
