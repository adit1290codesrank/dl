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
#include <cuda_runtime.h>

// SchemaRAGNet: Dual-Encoder with Cross-Attention into Schema
// Forward: query_encoder → cross_attn(Q, Schema) → residual → vocab_proj → raw logits
// The CE loss kernel internally handles softmax+CE combined gradient (pred - target).
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
            query_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth, true);  // Decoder (Causal)
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
            
            // Residual connection: lets gradients flow directly into query encoder
            context = context + Q_emb;
            
            int batch = context.shape[0];
            int seq = context.shape[1];
            
            context.shape = {batch * seq, dimension};
            Tensor logits = vocab_proj->forward(context);
            Tensor probs = sm->forward(logits);  // Softmax to get probabilities
            
            // CE loss needs probabilities: gradient = (prob - target), loss = -target*log(prob)
            // Softmax::backward() is identity, so no double-counting
            probs.shape = {batch, seq, vocab_size};
            return probs;
        }

        Tensor backward(const Tensor& dY, float lr)
        {
            int batch = dY.shape[0];
            int seq = dY.shape[1];
            
            Tensor flat_dY = dY;
            flat_dY.shape = {batch * seq, vocab_size};
            
            // sm->backward is identity — the CE gradient (pred-target) already accounts for softmax
            Tensor d = sm->backward(flat_dY, lr);
            d = vocab_proj->backward(d, lr);
            d.shape = {batch, seq, dimension};
            
            // Backprop through pointer attention (cross-attention)
            d = pointer_layer->backward(d, lr);
            
            // Backprop into schema encoder
            Tensor dSchema = pointer_layer->get_schema_grad();
            schema_encoder->backward(dSchema, lr);
            
            // Backprop into query encoder
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
                 const std::vector<float>& X_val, const std::vector<float>& Schema_val, const std::vector<float>& Y_val,
                 int n, int n_val, int seq_len, int schema_size, int vocab_size, int epochs, int bs, float lr)
        {
            set_mode(true);
            int nb = n / bs;
            if (nb == 0) nb = 1;

            Tensor loss_val = Tensor::zeros(1, 1);
            Tensor grad = Tensor::zeros(bs * seq_len, vocab_size);

            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::mt19937 rng(42);

            std::ofstream log_file("loss_log.csv");
            log_file << "epoch,train_loss,val_loss,top1_acc,top5_acc,lr\n";

            std::cout << "Starting SchemaRAG Training Loop..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();

            float lr_min = lr * 0.01f;

            for (int e = 1; e <= epochs; ++e)
            {
                std::shuffle(idx.begin(), idx.end(), rng);
                float tot_loss = 0.0f;
                
                float current_lr;
                int warmup_epochs = 10;
                if (e <= warmup_epochs) {
                    current_lr = lr * ((float)e / warmup_epochs);
                } else {
                    float progress = (float)(e - 1 - warmup_epochs) / (epochs - warmup_epochs);
                    current_lr = lr_min + 0.5f * (lr - lr_min) * (1.0f + std::cos(3.1415926535f * progress));
                }

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

                // Evaluate on validation set
                set_mode(false);
                int nb_val = n_val / bs;
                if (nb_val == 0) nb_val = 1;
                
                int top1_correct = 0;
                int top5_correct = 0;
                int total_valid_tokens = 0;
                float val_loss_sum = 0.0f;

                Tensor val_loss_tensor = Tensor::zeros(1, 1);

                for (int b = 0; b < nb_val; ++b) {
                    int actual_bs = std::min(bs, n_val - b * bs);
                    std::vector<float> bX(actual_bs * seq_len);
                    std::vector<float> bS(actual_bs * schema_size);
                    std::vector<float> bY(actual_bs * seq_len * vocab_size, 0.0f);

                    for (int i = 0; i < actual_bs; ++i) {
                        int id = b * bs + i;
                        std::copy(X_val.begin() + id * seq_len, X_val.begin() + (id + 1) * seq_len, bX.begin() + i * seq_len);
                        std::copy(Schema_val.begin() + id * schema_size, Schema_val.begin() + (id + 1) * schema_size, bS.begin() + i * schema_size);
                        for(int t = 0; t < seq_len; ++t) {
                            int token_id = (int)Y_val[id * seq_len + t];
                            if(token_id >= 0 && token_id < vocab_size) {
                                bY[(i * seq_len + t) * vocab_size + token_id] = 1.0f;
                            }
                        }
                    }

                    Tensor dX = Tensor::upload(bX, {actual_bs, seq_len});
                    Tensor dS = Tensor::upload(bS, {actual_bs, schema_size});
                    Tensor dY_val_t = Tensor::upload(bY, {actual_bs * seq_len, vocab_size});
                    
                    Tensor pred = forward(dX, dS);
                    pred.shape = {actual_bs * seq_len, vocab_size};
                    val_loss_sum += Loss::compute_loss(pred, dY_val_t, val_loss_tensor, LossType::CROSS_ENTROPY);
                    pred.shape = {actual_bs, seq_len, vocab_size};

                    std::vector<float> pred_cpu(pred.total_elements());
                    cudaMemcpy(pred_cpu.data(), pred.data(), pred_cpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

                    for (int i = 0; i < actual_bs; ++i) {
                        int id = b * bs + i;
                        for (int t = 0; t < seq_len; ++t) {
                            int target_token = (int)Y_val[id * seq_len + t];
                            if (target_token == 0) continue; // Skip padding

                            // Find top-5 tokens
                            int top5[5] = {-1, -1, -1, -1, -1};
                            float top5_vals[5] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
                            
                            for (int v = 0; v < vocab_size; ++v) {
                                float val = pred_cpu[(i * seq_len + t) * vocab_size + v];
                                // Insert into sorted top-5
                                for (int k = 0; k < 5; ++k) {
                                    if (val > top5_vals[k]) {
                                        // Shift down
                                        for (int j = 4; j > k; --j) {
                                            top5[j] = top5[j-1];
                                            top5_vals[j] = top5_vals[j-1];
                                        }
                                        top5[k] = v;
                                        top5_vals[k] = val;
                                        break;
                                    }
                                }
                            }
                            
                            if (top5[0] == target_token) top1_correct++;
                            for (int k = 0; k < 5; ++k) {
                                if (top5[k] == target_token) { top5_correct++; break; }
                            }
                            total_valid_tokens++;
                        }
                    }
                }
                
                float val_loss = val_loss_sum / nb_val;
                float top1_acc = total_valid_tokens > 0 ? (float)top1_correct / total_valid_tokens * 100.0f : 0.0f;
                float top5_acc = total_valid_tokens > 0 ? (float)top5_correct / total_valid_tokens * 100.0f : 0.0f;
                set_mode(true);

                std::cout << "\rEpoch " << e << "/" << epochs << "  Loss: " << avg << "  Val Loss: " << val_loss << "  Top1: " << top1_acc << "%  Top5: " << top5_acc << "%  LR: " << current_lr << std::endl;
                log_file << e << "," << avg << "," << val_loss << "," << top1_acc << "," << top5_acc << "," << current_lr << "\n";
                log_file.flush();
            }

            log_file.close();
            float secs = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "Done in " << secs << "s" << std::endl;
        }
};
