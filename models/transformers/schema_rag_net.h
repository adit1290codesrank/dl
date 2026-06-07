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

void clamp_tensor_cuda(float* data, float min_val, float max_val, int size);
void calculate_accuracy_cuda(const float* pred, const float* targets_idx, int* top1_out, int* top5_out, int* total_out, int seq_len, int vocab_size, int total_tokens);
Tensor matrix_multiply(const Tensor& A, bool transA, const Tensor& B, bool transB);
Tensor matrix_add(const Tensor& A, const Tensor& B);
void full_attention_softmax_backward_cuda(const float* dY, const float* Y, float* dX, int rows, int cols);

// Pointer / copy-head kernels (defined in cuda_ops.cu)
void reduce_heads_attn_cuda(const float* attn, float* out, int batch, int heads, int Tq, int S);
void apply_attention_mask_cuda(float* scores, const float* mask, int batch, int heads, int Tq, int S);
void expand_heads_attn_cuda(const float* d_mean, float* d_attn, int batch, int heads, int Tq, int S);
void copy_scatter_cuda(const float* attn_mean, const float* vocab_ids, float* P_schema, int batch_seq, int S, int V);
void copy_gather_cuda(const float* dP_schema, const float* vocab_ids, float* d_attn_mean, int batch_seq, int S, int V);
void blend_forward_cuda(const float* p_gen, const float* P_vocab, const float* P_schema, float* P_final, int batch_seq, int V);
void ce_prob_grad_cuda(const float* P_final, const float* targets_idx, float* dP, int batch_seq, int V, int valid_tokens);
void blend_backward_dist_cuda(const float* p_gen, const float* dP_final, float* dP_vocab, float* dP_schema, int batch_seq, int V);
void blend_backward_pgen_cuda(const float* P_vocab, const float* P_schema, const float* dP_final, float* dp_gen, int batch_seq, int V);
void sigmoid_forward_cuda(float* data, int size);
void sigmoid_grad_mul_cuda(const float* s, float* dy, int size);

// SchemaRAGNet: Dual-Encoder pointer-generator for text→SQL.
// Forward: query_encoder → cross_attn(Q, Schema) → residual → final_ln →
//   { tied vocab proj → softmax = P_vocab ; sigmoid gate p_gen ; schema attention scatter = P_schema }
//   → P_final = p_gen*P_vocab + (1-p_gen)*P_schema.
// Backward uses the true CE-on-probability gradient + real softmax Jacobian (no fused shortcut).
class SchemaRAGNet
{
    private:
        TextEncoder* query_encoder;
        TextEncoder* schema_encoder;
        PointerAttention* pointer_layer;
        LayerNorm* final_ln;
        Dense* p_gen_proj;   // Dense(dim, 1) → sigmoid = generate-vs-copy gate
        Softmax* sm;

        Tensor cached_proj_in;  // final_ln output (fed to both vocab proj and p_gen proj)
        Tensor cached_P_vocab;  // [B*T, V]
        Tensor cached_P_schema; // [B*T, V]
        Tensor cached_P_final;  // [B*T, V]
        Tensor cached_pgen;     // [B*T, 1]
        int cached_batch = 0, cached_seq = 0, cached_schema_size = 0;

        int vocab_size;
        int max_seq_len;
        int dimension;
        int heads;
        int depth;

        Tensor schema_vocab_ids; // [schema_size, 1] vocab id per schema slot (copy-head targets)
        Tensor schema_mask;      // [batch, schema_size] 1.0 for valid, 0.0 for pad

    public:
        SchemaRAGNet(int vocab_size, int max_seq_len, int dimension, int heads, int depth)
            : vocab_size(vocab_size), max_seq_len(max_seq_len), dimension(dimension), heads(heads), depth(depth)
        {
            query_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth, true);  // Decoder (Causal)
            schema_encoder = new TextEncoder(vocab_size, max_seq_len, dimension, heads, depth, false); // Encoder (Bidirectional)
            pointer_layer = new PointerAttention(dimension, heads);
            final_ln = new LayerNorm(dimension);
            p_gen_proj = new Dense(dimension, 1);
            sm = new Softmax();
        }

        ~SchemaRAGNet()
        {
            delete query_encoder;
            delete schema_encoder;
            delete pointer_layer;
            delete final_ln;
            delete p_gen_proj;
            delete sm;
        }

        void set_k_frozen(const Tensor& kf) {
            pointer_layer->set_k_frozen(kf);
        }

        void set_schema_vocab_ids(const Tensor& ids) {
            schema_vocab_ids = ids;
        }

        void set_schema_mask(const Tensor& mask) {
            schema_mask = mask;
        }

        // Returns the pointer-generator distribution P_final = p_gen*P_vocab + (1-p_gen)*P_schema.
        Tensor forward(const Tensor& query_tokens, const Tensor& schema_tokens) {
            Tensor Q_emb = query_encoder->forward(query_tokens);
            // schema_tokens: [batch, schema_size, max_schema_toks] → pooled to [batch, schema_size, dim]
            Tensor K_emb = schema_encoder->forward_pooled(schema_tokens);

            auto out = pointer_layer->forward_dual(Q_emb, K_emb, schema_mask);
            Tensor context = out.first;     // [batch, seq, dim]
            Tensor attn = out.second;       // [batch*heads, seq, schema_size]

            // Residual connection: lets gradients flow directly into query encoder
            context = context + Q_emb;

            int batch = context.shape[0];
            int seq = context.shape[1];
            int S = K_emb.shape[1];
            int BT = batch * seq;
            cached_batch = batch; cached_seq = seq; cached_schema_size = S;

            context.shape = {BT, dimension};
            Tensor ln_out = final_ln->forward(context);
            cached_proj_in = ln_out;

            // ---- Generate branch: tied vocab projection → softmax ----
            Tensor W_emb = query_encoder->token_embedding().weight();   // [vocab, dim]
            Tensor vocab_logits = matrix_multiply(ln_out, false, W_emb, true); // [BT, vocab]
            Tensor P_vocab = sm->forward(vocab_logits);
            cached_P_vocab = P_vocab;

            // ---- Copy gate p_gen ∈ (0,1) ----
            Tensor p_gen = p_gen_proj->forward(ln_out);   // [BT, 1]
            sigmoid_forward_cuda(p_gen.data(), p_gen.total_elements());
            cached_pgen = p_gen;

            // ---- Copy branch: attention over schema scattered into the vocab ----
            Tensor attn_mean(std::vector<int>{BT, S});
            reduce_heads_attn_cuda(attn.data(), attn_mean.data(), batch, heads, seq, S);
            Tensor P_schema(std::vector<int>{BT, vocab_size});
            copy_scatter_cuda(attn_mean.data(), schema_vocab_ids.data(), P_schema.data(), BT, S, vocab_size);
            cached_P_schema = P_schema;

            // ---- Blend ----
            Tensor P_final(std::vector<int>{BT, vocab_size});
            blend_forward_cuda(p_gen.data(), P_vocab.data(), P_schema.data(), P_final.data(), BT, vocab_size);
            cached_P_final = P_final;

            P_final.shape = {batch, seq, vocab_size};
            return P_final;
        }

        // Backward straight from the integer targets (CE on the mixture distribution).
        void backward_with_targets(const Tensor& targets_idx, int valid_tokens, float lr)
        {
            int batch = cached_batch, seq = cached_seq, S = cached_schema_size;
            int BT = batch * seq, V = vocab_size;

            // 1. True CE-on-probability gradient w.r.t. the mixture.
            Tensor dP_final(std::vector<int>{BT, V});
            ce_prob_grad_cuda(cached_P_final.data(), targets_idx.data(), dP_final.data(), BT, V, valid_tokens);

            // 2. Split across branches.
            Tensor dP_vocab(std::vector<int>{BT, V});
            Tensor dP_schema(std::vector<int>{BT, V});
            blend_backward_dist_cuda(cached_pgen.data(), dP_final.data(), dP_vocab.data(), dP_schema.data(), BT, V);

            Tensor dp_gen(std::vector<int>{BT, 1});
            blend_backward_pgen_cuda(cached_P_vocab.data(), cached_P_schema.data(), dP_final.data(), dp_gen.data(), BT, V);

            // 3. Generate branch: real softmax Jacobian → tied projection.
            Tensor dvocab_logits(std::vector<int>{BT, V});
            full_attention_softmax_backward_cuda(dP_vocab.data(), cached_P_vocab.data(), dvocab_logits.data(), BT, V);

            Tensor W_emb = query_encoder->token_embedding().weight();
            Tensor dW_emb = matrix_multiply(dvocab_logits, true, cached_proj_in, false); // [V, dim]
            query_encoder->token_embedding().add_external_grad(dW_emb);
            Tensor dctx_vocab = matrix_multiply(dvocab_logits, false, W_emb, false);      // [BT, dim]

            // 4. p_gen branch: sigmoid backward → Dense(dim,1).
            sigmoid_grad_mul_cuda(cached_pgen.data(), dp_gen.data(), BT); // dp_gen → d(logit)
            Tensor dctx_pgen = p_gen_proj->backward(dp_gen, lr);          // [BT, dim]

            // 5. Combine grads into ln_out, then through final LayerNorm.
            Tensor dln = matrix_add(dctx_vocab, dctx_pgen);
            Tensor dctx = final_ln->backward(dln, lr);                    // [BT, dim]
            dctx.shape = {batch, seq, dimension};

            // 6. Copy branch: gather → expand over heads → external attention gradient.
            Tensor d_attn_mean(std::vector<int>{BT, S});
            copy_gather_cuda(dP_schema.data(), schema_vocab_ids.data(), d_attn_mean.data(), BT, S, V);
            Tensor d_attn(std::vector<int>{batch * heads, seq, S});
            expand_heads_attn_cuda(d_attn_mean.data(), d_attn.data(), batch, heads, seq, S);

            // 7. Pointer backward (grad w.r.t. context_ptr = dctx; plus copy-attention grad).
            Tensor d_query = pointer_layer->backward_ext(dctx, lr, &d_attn);
            d_query.shape = {batch, seq, dimension};

            // 8. Residual: Q_emb received both the pointer-path grad and the direct context grad.
            Tensor dQ_emb = matrix_add(d_query, dctx);

            // 9. Schema encoder (pooled) and query encoder.
            Tensor dSchema = pointer_layer->get_schema_grad();
            schema_encoder->backward_pooled(dSchema, lr);
            query_encoder->backward(dQ_emb, lr);
        }

        void set_mode(bool train)
        {
            query_encoder->set_mode(train);
            schema_encoder->set_mode(train);
            pointer_layer->set_mode(train);
            final_ln->set_mode(train);
            p_gen_proj->set_mode(train);
            sm->set_mode(train);
        }

        void save(const std::string& path)
        {
            query_encoder->save(path + "_query.bin");
            schema_encoder->save(path + "_schema.bin");
            std::ofstream os(path + "_pointer.bin", std::ios::binary);
            pointer_layer->save(os);
            os.close();
            std::ofstream os_ln(path + "_ln.bin", std::ios::binary);
            final_ln->save(os_ln);
            os_ln.close();
            std::ofstream os_pg(path + "_pgen.bin", std::ios::binary);
            p_gen_proj->save(os_pg);
            os_pg.close();
            // vocab projection is tied to query_encoder.token_emb (saved with the query encoder).
        }

        void load(const std::string& path)
        {
            query_encoder->load(path + "_query.bin");
            schema_encoder->load(path + "_schema.bin");
            std::ifstream is(path + "_pointer.bin", std::ios::binary);
            pointer_layer->load(is);
            is.close();
            std::ifstream is_ln(path + "_ln.bin", std::ios::binary);
            if(is_ln.is_open()) {
                final_ln->load(is_ln);
                is_ln.close();
            }
            std::ifstream is_pg(path + "_pgen.bin", std::ios::binary);
            if(is_pg.is_open()) {
                p_gen_proj->load(is_pg);
                is_pg.close();
            }
            // vocab projection is tied to query_encoder.token_emb (loaded with the query encoder).
        }

        void fit(const std::vector<float>& X, const std::vector<float>& Schema, const std::vector<float>& Y,
                 const std::vector<float>& X_val, const std::vector<float>& Schema_val, const std::vector<float>& Y_val,
                 int n, int n_val, int seq_len, int schema_size, int max_schema_toks, int vocab_size, int epochs, int bs, float lr,
                 int warmup_override = -1)
        {
            set_mode(true);
            int nb = n / bs;
            if (nb == 0) nb = 1;

            int schema_stride = schema_size * max_schema_toks; // sub-tokens per example

            Tensor loss_val = Tensor::zeros(1, 1);

            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::mt19937 rng(42);

            std::ofstream log_file("loss_log.csv");
            log_file << "epoch,train_loss,val_loss,top1_acc,top5_acc,lr\n";

            std::cout << "Starting SchemaRAG Training Loop..." << std::endl;
            auto t0 = std::chrono::high_resolution_clock::now();

            float lr_min = lr * 0.01f;
            float best_top5 = -1.0f; // best validation top-5 seen so far (for checkpointing)

            for (int e = 1; e <= epochs; ++e)
            {
                std::shuffle(idx.begin(), idx.end(), rng);
                float tot_loss = 0.0f;
                
                float current_lr;
                // Default warmup = 10% of training; resume/fine-tune can pass a short override.
                int warmup_epochs = (warmup_override > 0) ? warmup_override : epochs / 10;
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
                    std::vector<float> bY_idx(actual_bs * seq_len);
                    std::vector<float> bS(actual_bs * schema_stride);
                    std::vector<float> bMask(actual_bs * schema_size);

                    int valid_tokens = 0;

                    for (int i = 0; i < actual_bs; ++i)
                    {
                        int id = idx[b * bs + i];
                        std::copy(X.begin() + id * seq_len, X.begin() + (id + 1) * seq_len, bX.begin() + i * seq_len);
                        std::copy(Schema.begin() + id * schema_stride, Schema.begin() + (id + 1) * schema_stride, bS.begin() + i * schema_stride);

                        for(int t = 0; t < seq_len; ++t) {
                            int token_id = (int)Y[id * seq_len + t];
                            bY_idx[i * seq_len + t] = (float)token_id;
                            if (token_id != -100) valid_tokens++;
                        }
                        
                        for (int s = 0; s < schema_size; ++s) {
                            int start = id * schema_stride + s * max_schema_toks;
                            float first_tok = Schema[start];
                            // If first token is PAD (0), mask is 0.0, else 1.0
                            bMask[i * schema_size + s] = (first_tok != 0.0f) ? 1.0f : 0.0f;
                        }
                    }
                    if (valid_tokens == 0) valid_tokens = 1;

                    Tensor tX = Tensor::upload(bX, {actual_bs, seq_len});
                    Tensor tY_idx = Tensor::upload(bY_idx, {actual_bs * seq_len, 1});
                    Tensor tS = Tensor::upload(bS, {actual_bs, schema_size, max_schema_toks});
                    Tensor tMask = Tensor::upload(bMask, {actual_bs, schema_size});
                    set_schema_mask(tMask);

                    Tensor pred = forward(tX, tS);   // P_final [bs, seq, vocab]

                    // Backward computes the CE-on-mixture gradient internally from the targets.
                    backward_with_targets(tY_idx, valid_tokens, current_lr);

                    // Loss is monitored on the mixture distribution (valid for any prob dist).
                    pred.shape = {actual_bs * seq_len, vocab_size};
                    tot_loss += Loss::compute_loss(pred, dY, loss_val, LossType::CROSS_ENTROPY, valid_tokens);
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
                    std::vector<float> bS(actual_bs * schema_stride);
                    std::vector<float> bMask(actual_bs * schema_size);

                    for (int i = 0; i < actual_bs; ++i) {
                        int id = b * bs + i;
                        std::copy(X_val.begin() + id * seq_len, X_val.begin() + (id + 1) * seq_len, bX.begin() + i * seq_len);
                        std::copy(Schema_val.begin() + id * schema_stride, Schema_val.begin() + (id + 1) * schema_stride, bS.begin() + i * schema_stride);
                        
                        for (int s = 0; s < schema_size; ++s) {
                            int start = id * schema_stride + s * max_schema_toks;
                            float first_tok = Schema_val[start];
                            bMask[i * schema_size + s] = (first_tok != 0.0f) ? 1.0f : 0.0f;
                        }
                    }

                    Tensor dX = Tensor::upload(bX, {actual_bs, seq_len});
                    Tensor dS = Tensor::upload(bS, {actual_bs, schema_size, max_schema_toks});
                    Tensor dMask = Tensor::upload(bMask, {actual_bs, schema_size});
                    set_schema_mask(dMask);
                    
                    std::vector<float> bY_idx(actual_bs * seq_len);
                    int valid_tokens = 0;
                    for (int i = 0; i < actual_bs; ++i) {
                        int id = b * bs + i;
                        for(int t = 0; t < seq_len; ++t) {
                            int token_id = (int)Y_val[id * seq_len + t];
                            bY_idx[i * seq_len + t] = (float)token_id;
                            if (token_id != -100) valid_tokens++;
                        }
                    }
                    if (valid_tokens == 0) valid_tokens = 1;
                    Tensor dY_idx = Tensor::upload(bY_idx, {actual_bs * seq_len, 1});
                    
                    Tensor pred = forward(dX, dS);
                    pred.shape = {actual_bs * seq_len, vocab_size};
                    val_loss_sum += Loss::compute_loss(pred, dY_idx, val_loss_tensor, LossType::CROSS_ENTROPY, valid_tokens);
                    
                    int b_top1, b_top5, b_total;
                    calculate_accuracy_cuda(pred.data(), dY_idx.data(), &b_top1, &b_top5, &b_total, seq_len, vocab_size, actual_bs * seq_len);
                    
                    top1_correct += b_top1;
                    top5_correct += b_top5;
                    total_valid_tokens += b_total;

                    pred.shape = {actual_bs, seq_len, vocab_size};
                }
                
                float val_loss = val_loss_sum / nb_val;
                float top1_acc = total_valid_tokens > 0 ? (float)top1_correct / total_valid_tokens * 100.0f : 0.0f;
                float top5_acc = total_valid_tokens > 0 ? (float)top5_correct / total_valid_tokens * 100.0f : 0.0f;
                set_mode(true);

                std::cout << "\rEpoch " << e << "/" << epochs << "  Loss: " << avg << "  Val Loss: " << val_loss << "  Top1: " << top1_acc << "%  Top5: " << top5_acc << "%  LR: " << current_lr << std::endl;
                log_file << e << "," << avg << "," << val_loss << "," << top1_acc << "," << top5_acc << "," << current_lr << "\n";
                log_file.flush();

                // Checkpoint the best-so-far model so the run can be stopped at any time.
                // The standard weights path always holds the best validation top-5 checkpoint.
                if (top5_acc > best_top5) {
                    best_top5 = top5_acc;
                    save("weights/schema_rag.bin");
                    std::cout << "  [checkpoint] new best Top5 " << top5_acc << "% -> weights/schema_rag.bin" << std::endl;
                }
            }

            log_file.close();
            float secs = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "Done in " << secs << "s" << std::endl;
        }
};
