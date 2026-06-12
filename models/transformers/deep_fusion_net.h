#pragma once
#include "../../include/core/tensor.h"
#include "../../include/core/loss.h"
#include "../../include/layers/decoder_block.h"
#include "../../include/layers/embedding.h"
#include "../../include/layers/dense.h"
#include "../../include/layers/softmax.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>

// Kernels from cuda_ops.cu / elsewhere
Tensor matrix_multiply(const Tensor& A, bool transA, const Tensor& B, bool transB);
Tensor matrix_add(const Tensor& A, const Tensor& B);
void full_attention_softmax_backward_cuda(const float* dY, const float* Y, float* dX, int rows, int cols);
void copy_scatter_cuda(const float* attn_mean, const float* vocab_ids, float* P_schema, int batch_seq, int S, int V);
void copy_gather_cuda(const float* dP_schema, const float* vocab_ids, float* d_attn_mean, int batch_seq, int S, int V);
void blend_forward_cuda(const float* p_gen, const float* P_vocab, const float* P_schema, float* P_final, int batch_seq, int V);
void blend_backward_dist_cuda(const float* p_gen, const float* dP_final, float* dP_vocab, float* dP_schema, int batch_seq, int V);
void ce_prob_grad_eps_cuda(const float* P_final, const float* targets_idx, float* dP, int batch_seq, int V, int valid_tokens, float eps);
void blend_backward_pgen_exact_cuda(const float* dP_final, const float* P_vocab, const float* P_mem, const float* targets_idx, float* dp_gen, int batch_seq, int V);
void sigmoid_forward_cuda(float* data, int size);
void sigmoid_grad_mul_cuda(const float* s, float* dy, int size);
void mask_generator_logits_cuda(float* logits, int batch_seq, int vocab_size, int bpe_vocab_size);
void attention_scale_cuda(float* data, float scale, int size);
void mean_pool_groups_cuda(const float* in, const float* tokens, float* out, int num_groups, int group_size, int dim, float pad_id);
void mean_pool_groups_backward_cuda(const float* dout, const float* tokens, float* din, int num_groups, int group_size, int dim, float pad_id);
void embedding_backward_cuda(const float* X, const float* dY, float* dW, int tokens, int dimension, int size);
void calculate_accuracy_cuda(const float* pred, const float* targets_idx, int* top1_out, int* top5_out, int* total_out, int seq_len, int vocab_size, int total_tokens);

// DeepFusionNet: C++ port of scripts/schema_fusion_pt.py::DeepFusionNet.
//
// Single causal decoder over [Question][SEP][SQL] with cross-attention into a
// GLOBAL memory bank (schema columns + synonyms + jargon fragments) in EVERY
// block. Output head is a pointer-generator over a compact vocab:
//   [0, V_bpe)  real BPE tokens via the (tied-embedding) generator softmax,
//   [V_bpe, V)  atomic schema/fragment ids via a DEDICATED pointer head
//               (own Q/K projections over the pooled memory keys -- decoupled
//               from the in-block context attention).
//   P_final = p_gen * P_vocab + (1 - p_gen) * P_mem
// Memory keys = masked mean-pool of each row's sub-token embeddings + a type
// embedding (0 table / 1 column / 2 fragment). One token embedding is shared
// by the input sequence, the memory keys, and the output projection; its
// three gradient sources are folded into a single Adam step.
class DeepFusionNet
{
    private:
        int V, V_bpe, max_seq_len, dimension, heads, depth;

        Embedding token_emb;   // [V, dim] -- shared 3 ways
        Embedding pos_emb;     // [max_seq_len, dim]
        Embedding type_emb;    // [3, dim]
        std::vector<std::unique_ptr<DecoderBlock>> blocks;
        LayerNorm final_ln;
        Dense p_gen_proj;      // dim -> 1, sigmoid gate
        Dense ptr_q, ptr_k;    // dedicated pointer head projections
        Softmax sm;

        // Global memory bank (set once)
        Tensor mem_tokens;     // [M, gsize] sub-token ids
        Tensor mem_types;      // [1, M] row types
        Tensor mem_emit_ids;   // [M, 1] vocab id each row emits
        int M = 0, gsize = 0;

        // Forward caches for backward
        Tensor cached_mem_flat;   // [1, M*gsize]
        Tensor cached_mem_pooled; // [M, dim]
        Tensor cached_ln_out;     // [BT, dim]
        Tensor cached_P_vocab, cached_P_mem, cached_P_final; // [BT, V]
        Tensor cached_P_ptr;      // [BT, M]
        Tensor cached_pgen;       // [BT, 1]
        Tensor cached_q;          // [BT, dim]
        Tensor cached_k;          // [M, dim]
        int cached_batch = 0, cached_seq = 0;

        void init_embedding_(Embedding& e, int rows, unsigned seed)
        {
            // GPT-style std=0.02 init, matching the PyTorch reference. The
            // framework default (std = sqrt(2/dim)) explodes the tied-softmax
            // logits and saturates training (learned the hard way in Phase A).
            std::mt19937 gen(seed);
            std::normal_distribution<float> dist(0.0f, 0.02f);
            std::vector<float> w(rows * dimension);
            for (auto& v : w) v = dist(gen);
            e.weight() = Tensor::upload(w, rows, dimension);
        }

    public:
        DeepFusionNet(int V, int V_bpe, int max_seq_len, int dimension, int heads, int depth, float dropout)
            : V(V), V_bpe(V_bpe), max_seq_len(max_seq_len), dimension(dimension), heads(heads), depth(depth),
              token_emb(V, dimension), pos_emb(max_seq_len, dimension), type_emb(3, dimension),
              final_ln(dimension), p_gen_proj(dimension, 1),
              ptr_q(dimension, dimension), ptr_k(dimension, dimension)
        {
            for (int i = 0; i < depth; ++i)
                blocks.push_back(std::make_unique<DecoderBlock>(dimension, heads, dropout));

            init_embedding_(token_emb, V, 42);
            init_embedding_(pos_emb, max_seq_len, 43);
            init_embedding_(type_emb, 3, 44);

            // p_gen bias 2.5 keeps the gate ~0.92 early so the flat generator
            // distribution isn't drowned by the peaked pointer distribution.
            std::vector<float> b_init = {2.5f};
            p_gen_proj.set_bias(Tensor::upload(b_init, {1, 1}));
        }

        void set_memory(const Tensor& tokens, const Tensor& types, const Tensor& emit_ids)
        {
            mem_tokens = tokens;       // [M, gsize]
            M = tokens.shape[0];
            gsize = tokens.shape[1];
            mem_types = types;         // [1, M]
            mem_emit_ids = emit_ids;   // [M, 1]
        }

        // Encode the memory bank: pooled sub-token embeddings + type embedding.
        Tensor encode_memory()
        {
            cached_mem_flat = mem_tokens.reshape({1, M * gsize});
            Tensor emb = token_emb.forward(cached_mem_flat);   // [1, M*gsize, dim]

            Tensor pooled(std::vector<int>{M, dimension});
            mean_pool_groups_cuda(emb.data(), cached_mem_flat.data(), pooled.data(), M, gsize, dimension, 0.0f);

            Tensor tv = type_emb.forward(mem_types);           // [1, M, dim]
            tv.shape = {M, dimension};
            return matrix_add(pooled, tv);                     // [M, dim]
        }

        // X: [batch, seq] token ids. Returns P_final [batch, seq, V].
        Tensor forward(const Tensor& X)
        {
            int batch = X.shape[0];
            int seq = X.shape[1];
            int BT = batch * seq;
            cached_batch = batch; cached_seq = seq;

            // ---- memory bank (encoded BEFORE the sequence so token_emb's
            // lookup cache ends holding X for the sequence-path backward) ----
            cached_mem_pooled = encode_memory();

            // ---- input embedding + learned positions ----
            Tensor x = token_emb.forward(X);                   // [B, T, dim]
            std::vector<float> pidx(BT);
            for (int b = 0; b < batch; ++b)
                for (int t = 0; t < seq; ++t)
                    pidx[b * seq + t] = (float)(t % max_seq_len);
            x = x + pos_emb.forward(Tensor::upload(pidx, {batch, seq}));

            // ---- broadcast memory to the batch + all-valid mask ----
            Tensor mem_b(std::vector<int>{batch, M, dimension});
            size_t row_bytes = (size_t)M * dimension * sizeof(float);
            for (int b = 0; b < batch; ++b)
                cudaMemcpy(mem_b.data() + (size_t)b * M * dimension, cached_mem_pooled.data(), row_bytes, cudaMemcpyDeviceToDevice);
            Tensor mem_mask = Tensor::upload(std::vector<float>(batch * M, 1.0f), {batch, M});

            // ---- decoder blocks (deep fusion: cross-attn in every block) ----
            for (auto& blk : blocks)
                x = blk->forward(x, mem_b, mem_mask);

            x.shape = {BT, dimension};
            Tensor ln_out = final_ln.forward(x);
            cached_ln_out = ln_out;

            // ---- generator branch: tied vocab projection, atomic ids masked ----
            Tensor W_emb = token_emb.weight();                            // [V, dim]
            Tensor vocab_logits = matrix_multiply(ln_out, false, W_emb, true); // [BT, V]
            mask_generator_logits_cuda(vocab_logits.data(), BT, V, V_bpe);
            Tensor P_vocab = sm.forward(vocab_logits);
            cached_P_vocab = P_vocab;

            // ---- copy gate ----
            Tensor p_gen = p_gen_proj.forward(ln_out);                    // [BT, 1]
            sigmoid_forward_cuda(p_gen.data(), p_gen.total_elements());
            cached_pgen = p_gen;

            // ---- dedicated pointer head ----
            Tensor q = ptr_q.forward(ln_out);                             // [BT, dim]
            Tensor k = ptr_k.forward(cached_mem_pooled);                  // [M, dim]
            cached_q = q; cached_k = k;
            Tensor ptr_logits = matrix_multiply(q, false, k, true);       // [BT, M]
            attention_scale_cuda(ptr_logits.data(), 1.0f / std::sqrt((float)dimension), BT * M);
            Tensor P_ptr = sm.forward(ptr_logits);
            cached_P_ptr = P_ptr;

            // scatter pointer mass into vocab space (atomicAdd: zero first;
            // multiple rows -- synonyms/jargon keys -- may share an emit id)
            Tensor P_mem(std::vector<int>{BT, V});
            P_mem.zero_();
            copy_scatter_cuda(P_ptr.data(), mem_emit_ids.data(), P_mem.data(), BT, M, V);
            cached_P_mem = P_mem;

            // ---- blend ----
            Tensor P_final(std::vector<int>{BT, V});
            blend_forward_cuda(p_gen.data(), P_vocab.data(), P_mem.data(), P_final.data(), BT, V);
            cached_P_final = P_final;

            P_final.shape = {batch, seq, V};
            return P_final;
        }

        void backward_with_targets(const Tensor& targets_idx, int valid_tokens, float lr)
        {
            int batch = cached_batch, seq = cached_seq;
            int BT = batch * seq;

            // 1. True CE-on-probability gradient (NO label smoothing: parity
            //    with the PyTorch reference's NLLLoss).
            Tensor dP_final(std::vector<int>{BT, V});
            ce_prob_grad_eps_cuda(cached_P_final.data(), targets_idx.data(), dP_final.data(), BT, V, valid_tokens, 0.0f);

            // 2. Split the blend.
            Tensor dP_vocab(std::vector<int>{BT, V});
            Tensor dP_mem(std::vector<int>{BT, V});
            blend_backward_dist_cuda(cached_pgen.data(), dP_final.data(), dP_vocab.data(), dP_mem.data(), BT, V);

            // 3. Generator softmax Jacobian.
            Tensor dvocab_logits(std::vector<int>{BT, V});
            full_attention_softmax_backward_cuda(dP_vocab.data(), cached_P_vocab.data(), dvocab_logits.data(), BT, V);

            // 4. Pointer head backward: gather -> softmax Jacobian -> scale ->
            //    matmuls -> ptr_q / ptr_k.
            Tensor dP_ptr(std::vector<int>{BT, M});
            copy_gather_cuda(dP_mem.data(), mem_emit_ids.data(), dP_ptr.data(), BT, M, V);
            Tensor d_ptr_logits(std::vector<int>{BT, M});
            full_attention_softmax_backward_cuda(dP_ptr.data(), cached_P_ptr.data(), d_ptr_logits.data(), BT, M);
            attention_scale_cuda(d_ptr_logits.data(), 1.0f / std::sqrt((float)dimension), BT * M);

            Tensor dq = matrix_multiply(d_ptr_logits, false, cached_k, false); // [BT, dim]
            Tensor dk = matrix_multiply(d_ptr_logits, true, cached_q, false);  // [M, dim]
            Tensor dln_ptr = ptr_q.backward(dq, lr);                           // [BT, dim]
            Tensor d_mem = ptr_k.backward(dk, lr);                             // [M, dim]

            // 5. Exact gate gradient, then sigmoid + Dense backward.
            Tensor dp_gen(std::vector<int>{BT, 1});
            blend_backward_pgen_exact_cuda(dP_final.data(), cached_P_vocab.data(), cached_P_mem.data(), targets_idx.data(), dp_gen.data(), BT, V);
            sigmoid_grad_mul_cuda(cached_pgen.data(), dp_gen.data(), BT);
            Tensor dln_pgen = p_gen_proj.backward(dp_gen, lr);                 // [BT, dim]

            // 6. Tied projection: stash dW for the single combined Adam step;
            //    context grad flows back through W_emb.
            Tensor W_emb = token_emb.weight();
            Tensor dW_tied = matrix_multiply(dvocab_logits, true, cached_ln_out, false); // [V, dim]
            Tensor dctx_vocab = matrix_multiply(dvocab_logits, false, W_emb, false);     // [BT, dim]

            // 7. Combine the three head gradients through the final LayerNorm.
            Tensor dln = matrix_add(matrix_add(dctx_vocab, dln_ptr), dln_pgen);
            Tensor dx = final_ln.backward(dln, lr);
            dx.shape = {batch, seq, dimension};

            // 8. Blocks in reverse; accumulate each block's memory gradient
            //    (sum over the batch broadcast: ones [1,B] @ [B, M*dim]).
            Tensor ones_b = Tensor::upload(std::vector<float>(batch, 1.0f), {1, batch});
            for (int i = depth - 1; i >= 0; --i) {
                dx = blocks[i]->backward(dx, lr);
                Tensor dSchema = blocks[i]->get_mem_grad();        // [B, M, dim]
                dSchema.shape = {batch, M * dimension};
                Tensor summed = matrix_multiply(ones_b, false, dSchema, false); // [1, M*dim]
                summed.shape = {M, dimension};
                d_mem = matrix_add(d_mem, summed);
            }

            // 9. Memory path: type embedding gets d_mem directly; the token
            //    side un-pools and is scattered into a full [V, dim] grad that
            //    joins the tied-projection grad in ONE external stash.
            type_emb.backward(d_mem, lr);

            Tensor d_unpooled(std::vector<int>{M * gsize, dimension});
            mean_pool_groups_backward_cuda(d_mem.data(), cached_mem_flat.data(), d_unpooled.data(), M, gsize, dimension, 0.0f);
            Tensor dW_mem = Tensor::zeros(V, dimension);
            embedding_backward_cuda(cached_mem_flat.data(), d_unpooled.data(), dW_mem.data(), M * gsize, dimension, V);
            token_emb.add_external_grad(matrix_add(dW_tied, dW_mem));

            // 10. Sequence path: positions, then the token embedding (whose
            //     backward folds in the external grad -> one Adam step).
            pos_emb.backward(dx, lr);
            token_emb.backward(dx, lr);
        }

        void set_mode(bool train)
        {
            for (auto& b : blocks) b->set_mode(train);
            final_ln.set_mode(train);
            p_gen_proj.set_mode(train);
            ptr_q.set_mode(train); ptr_k.set_mode(train);
            token_emb.set_mode(train); pos_emb.set_mode(train); type_emb.set_mode(train);
            sm.set_mode(train);
        }

        void save(const std::string& path)
        {
            std::ofstream os(path, std::ios::binary);
            if (!os.is_open()) throw std::runtime_error("Could not open " + path);
            token_emb.save(os); pos_emb.save(os); type_emb.save(os);
            for (auto& b : blocks) b->save(os);
            final_ln.save(os); p_gen_proj.save(os); ptr_q.save(os); ptr_k.save(os);
            os.close();
        }

        void load(const std::string& path)
        {
            std::ifstream is(path, std::ios::binary);
            if (!is.is_open()) throw std::runtime_error("Could not open " + path);
            token_emb.load(is); pos_emb.load(is); type_emb.load(is);
            for (auto& b : blocks) b->load(is);
            final_ln.load(is); p_gen_proj.load(is); ptr_q.load(is); ptr_k.load(is);
            is.close();
        }

        // Training loop: mirrors schema_rag_net::fit but with a GLOBAL memory
        // bank (no per-example schema copies) and the fusion.bin layout.
        void fit(const std::vector<float>& X, const std::vector<float>& Y,
                 const std::vector<float>& X_val, const std::vector<float>& Y_val,
                 int n, int n_val, int seq_len, int epochs, int bs, float peak_lr, int warmup_epochs)
        {
            int nb = n / bs; if (nb == 0) nb = 1;
            Tensor loss_val = Tensor::zeros(1, 1);

            std::vector<int> idx(n);
            std::iota(idx.begin(), idx.end(), 0);
            std::mt19937 rng(42);

            std::ofstream log_file("loss_log_fusion_cpp.csv");
            log_file << "epoch,train_loss,val_loss,top1,top5,lr\n";

            float lr_min = peak_lr * 0.01f;
            float best_top5 = -1.0f;
            auto t_start = std::chrono::high_resolution_clock::now();

            for (int e = 1; e <= epochs; ++e)
            {
                set_mode(true);
                std::shuffle(idx.begin(), idx.end(), rng);

                float current_lr;
                if (e <= warmup_epochs) {
                    current_lr = 1e-6f + (peak_lr - 1e-6f) * ((float)e / warmup_epochs);
                } else {
                    float progress = (float)(e - warmup_epochs) / std::max(1, epochs - warmup_epochs);
                    current_lr = lr_min + 0.5f * (peak_lr - lr_min) * (1.0f + std::cos(3.1415926535f * progress));
                }

                float tot_loss = 0.0f;
                for (int b = 0; b < nb; ++b)
                {
                    int actual_bs = std::min(bs, n - b * bs);
                    std::vector<float> bX(actual_bs * seq_len), bY(actual_bs * seq_len);
                    int valid_tokens = 0;
                    for (int i = 0; i < actual_bs; ++i) {
                        int id = idx[b * bs + i];
                        std::copy(X.begin() + (size_t)id * seq_len, X.begin() + (size_t)(id + 1) * seq_len, bX.begin() + (size_t)i * seq_len);
                        for (int t = 0; t < seq_len; ++t) {
                            int tok = (int)Y[(size_t)id * seq_len + t];
                            bY[(size_t)i * seq_len + t] = (float)tok;
                            if (tok != -100) valid_tokens++;
                        }
                    }
                    if (valid_tokens == 0) valid_tokens = 1;

                    Tensor tX = Tensor::upload(bX, {actual_bs, seq_len});
                    Tensor tY = Tensor::upload(bY, {actual_bs * seq_len, 1});

                    Tensor pred = forward(tX);
                    backward_with_targets(tY, valid_tokens, current_lr);

                    pred.shape = {actual_bs * seq_len, V};
                    tot_loss += Loss::compute_loss(pred, tY, loss_val, LossType::CROSS_ENTROPY, valid_tokens);

                    if (b % 5 == 0 || b == nb - 1)
                        std::cout << "\rEpoch " << e << "/" << epochs << "  batch " << b + 1 << "/" << nb << std::flush;
                }
                float train_loss = tot_loss / nb;

                // ---- validation (teacher-forced loss + top1/top5) ----
                set_mode(false);
                int nb_val = n_val / bs; if (nb_val == 0) nb_val = 1;
                float val_loss = 0.0f;
                long long c1 = 0, c5 = 0, ctot = 0;
                for (int b = 0; b < nb_val; ++b)
                {
                    int actual_bs = std::min(bs, n_val - b * bs);
                    std::vector<float> bX(actual_bs * seq_len), bY(actual_bs * seq_len);
                    int valid_tokens = 0;
                    for (int i = 0; i < actual_bs; ++i) {
                        int id = b * bs + i;
                        std::copy(X_val.begin() + (size_t)id * seq_len, X_val.begin() + (size_t)(id + 1) * seq_len, bX.begin() + (size_t)i * seq_len);
                        for (int t = 0; t < seq_len; ++t) {
                            int tok = (int)Y_val[(size_t)id * seq_len + t];
                            bY[(size_t)i * seq_len + t] = (float)tok;
                            if (tok != -100) valid_tokens++;
                        }
                    }
                    if (valid_tokens == 0) valid_tokens = 1;

                    Tensor tX = Tensor::upload(bX, {actual_bs, seq_len});
                    Tensor tY = Tensor::upload(bY, {actual_bs * seq_len, 1});
                    Tensor pred = forward(tX);
                    pred.shape = {actual_bs * seq_len, V};
                    val_loss += Loss::compute_loss(pred, tY, loss_val, LossType::CROSS_ENTROPY, valid_tokens);

                    int t1 = 0, t5 = 0, tt = 0;
                    calculate_accuracy_cuda(pred.data(), tY.data(), &t1, &t5, &tt, seq_len, V, actual_bs * seq_len);
                    c1 += t1; c5 += t5; ctot += tt;
                }
                val_loss /= nb_val;
                float top1 = ctot ? 100.0f * c1 / ctot : 0.0f;
                float top5 = ctot ? 100.0f * c5 / ctot : 0.0f;

                auto t_now = std::chrono::high_resolution_clock::now();
                float secs = std::chrono::duration<float>(t_now - t_start).count();
                std::cout << "\rEpoch " << e << "/" << epochs
                          << " | Loss: " << train_loss
                          << " | Val Loss: " << val_loss
                          << " | Top1: " << top1 << "%"
                          << " | Top5: " << top5 << "%"
                          << " | LR: " << current_lr
                          << " | t=" << (int)secs << "s" << std::endl;
                log_file << e << "," << train_loss << "," << val_loss << ","
                         << top1 << "," << top5 << "," << current_lr << "\n";
                log_file.flush();

                if (top5 > best_top5) {
                    best_top5 = top5;
                    save("weights/deep_fusion.bin");
                    std::cout << "    [checkpoint] new best Top5 " << top5 << "% -> weights/deep_fusion.bin" << std::endl;
                }
            }
        }
};
