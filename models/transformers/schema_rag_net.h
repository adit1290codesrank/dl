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
            query_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth, true); // Decoder (Causal)
            schema_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth, false); // Encoder (Bidirectional)
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

        void set_k_frozen(const Tensor& kf) {
            pointer_layer->set_k_frozen(kf);
        }

        Tensor forward(const Tensor& query_tokens, const Tensor& schema_tokens) {
            Tensor Q_emb = query_encoder->forward(query_tokens);
            Tensor K_emb = schema_encoder->forward(schema_tokens);
            
            auto out = pointer_layer->forward_dual(Q_emb, K_emb);
            Tensor context = out.first;
            // pointer_weights = out.second; // if needed later
            
            // CRITICAL FIX: Add residual connection so gradients can flow directly 
            // into the Query Encoder, bypassing the frozen Schema vectors!
            context = context + Q_emb;
            
            // Note: pointer scatter logic omitted for backprop simplicity
            // Standard Seq2Seq Projection
            int batch = context.shape[0];
            int seq = context.shape[1];
            
            context.shape = {batch * seq, dimension};
            Tensor logits = vocab_proj->forward(context);
            context.shape = {batch, seq, dimension};
            
            Tensor final_out = sm->forward(logits);
            final_out.shape = {batch, seq, vocab_size};
            return final_out;
        }

        Tensor backward(const Tensor& dY, float lr)
        {
            int batch = dY.shape[0];
            int seq = dY.shape[1];
            
            Tensor flat_dY = dY;
            flat_dY.shape = {batch * seq, vocab_size};
            
            Tensor d = sm->backward(flat_dY, lr);
            d = vocab_proj->backward(d, lr);
            
            d.shape = {batch, seq, dimension};
            
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
            Tensor grad = Tensor::zeros({bs * seq_len, vocab_size});

            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::mt19937 rng(42);

            std::ofstream log_file("loss_log.csv");
            log_file << "epoch,loss,lr\n";

            std::cout << "Starting SchemaRAG Training Loop..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();

            float lr_min = lr * 0.01f;

            for (int e = 1; e <= epochs; ++e)
            {
                std::shuffle(idx.begin(), idx.end(), rng);
                float tot_loss = 0.0f;
                
                float current_lr = lr_min + 0.5f * (lr - lr_min) * (1.0f + std::cos(3.1415926535f * (e - 1) / epochs));

                for (int b = 0; b < nb; ++b)
                {
                    int actual_bs = std::min(bs, n - b * bs);
                    std::vector<float> bX(actual_bs * seq_len);
                    std::vector<float> bY(actual_bs * seq_len * vocab_size, 0.0f);
                    std::vector<float> bS(actual_bs * schema_size);

                    for (int i = 0; i < actual_bs; ++i)
                    {
                        int id = idx[b * bs + i];
                        std::copy(X.begin() + id * seq_len, X.begin() + (id + 1) * seq_len, bX.begin() + i * seq_len);
                        std::copy(Schema.begin() + id * schema_size, Schema.begin() + (id + 1) * schema_size, bS.begin() + i * schema_size);
                        
                        for(int t = 0; t < seq_len; ++t) {
                            int token_id = (int)Y[id * seq_len + t];
                            if(token_id >= 0 && token_id < vocab_size) {
                                bY[(i * seq_len + t) * vocab_size + token_id] = 1.0f;
                            }
                        }
                    }

                    Tensor dX = Tensor::upload(bX, {actual_bs, seq_len});
                    Tensor dS = Tensor::upload(bS, {actual_bs, schema_size});
                    Tensor dY = Tensor::upload(bY, {actual_bs * seq_len, vocab_size});

                    Tensor pred = forward(dX, dS);
                    
                    // Reshape to flat 2D for loss calculation
                    pred.shape = {actual_bs * seq_len, vocab_size};
                    
                    Loss::compute_gradient(pred, dY, grad, LossType::CROSS_ENTROPY);
                    grad.shape = {actual_bs, seq_len, vocab_size};
                    backward(grad, current_lr);
                    tot_loss += Loss::compute_loss(pred, dY, loss_val, LossType::CROSS_ENTROPY);
                    
                    pred.shape = {actual_bs, seq_len, vocab_size}; // restore shape

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
                std::cout << "\rEpoch " << e << "/" << epochs << "  Loss: " << avg << "  LR: " << current_lr << "                            " << std::endl;
                log_file << e << "," << avg << "," << current_lr << "\n";
                log_file.flush(); // Flush buffer to disk immediately so you can monitor it live!
            }

            log_file.close();
            float secs = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "Done in " << secs << "s" << std::endl;
        }
};
