#pragma once
#include "../../include/core/tensor.h"
#include "../../include/layers/layer.h"
#include "../../include/layers/embedding.h"
#include "../../include/layers/transformer.h"
#include "../../include/layers/dropout.h"
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>

void mean_pool_groups_cuda(const float* in, const float* tokens, float* out, int num_groups, int group_size, int dim, float pad_id);
void mean_pool_groups_backward_cuda(const float* dout, const float* tokens, float* din, int num_groups, int group_size, int dim, float pad_id);

/*
Pure Sequence Encoder for Text.

Shape flow (forward):
  Input:           {batch, seq_len}          token IDs
  token_emb:       {batch, seq_len, dim}     embedding lookup appends dim
  pos_emb:         {batch, seq_len, dim}     same
  x = tok + pos:   {batch, seq_len, dim}     3D
  Transformer[]:   {batch, seq_len, dim}     each block preserves 3D shape
  Output:          {batch, seq_len, dim}     Raw sequence representations
*/

class TextEncoder : public Layer
{
    private:
        int vocab_size, seq_len, dim, depth;

        Embedding token_emb;
        Embedding pos_emb;
        Dropout emb_drop;
        std::vector<std::unique_ptr<Transformer>> blocks;

        // State for forward_pooled / backward_pooled
        int cached_pool_groups = 0;
        int cached_pool_gsize = 0;
        Tensor cached_pool_tokens; // flat [batch, n_elem*sub_toks] token ids, for masked pooling

    public:
        Embedding& token_embedding() { return token_emb; }
        TextEncoder(int vocab_size, int max_len, int dim, int heads, int depth, bool causal = false)
            : vocab_size(vocab_size), seq_len(max_len), dim(dim), depth(depth),
              token_emb(vocab_size, dim), pos_emb(max_len, dim), emb_drop(0.1f)
        {
            for(int i = 0; i < depth; ++i) 
                blocks.push_back(std::make_unique<Transformer>(dim, heads, causal));
        }

        Tensor forward(const Tensor& X) override
        {
            int batch = X.shape[0];
            int cur_seq_len = X.shape[1];

            Tensor x = token_emb.forward(X);

            std::vector<float> pidx(batch * cur_seq_len);
            for(int b = 0; b < batch; ++b) {
                for(int t = 0; t < cur_seq_len; ++t) {
                    pidx[b * cur_seq_len + t] = (float)(t % seq_len); // modulo prevents bounds error if schema is longer
                }
            }
            
            x = x + pos_emb.forward(Tensor::upload(pidx, {batch, cur_seq_len}));
            x = emb_drop.forward(x);

            for(auto& blk : blocks) {
                x = blk->forward(x);
            }
            
            // Return raw 3D tensor: [batch, seq_len, dim]
            return x;
        }

        Tensor backward(const Tensor& dY, float lr) override
        {
            Tensor d_enc = dY;
            
            for(int i = depth - 1; i >= 0; --i) {
                d_enc = blocks[i]->backward(d_enc, lr);
            }

            d_enc = emb_drop.backward(d_enc, lr);

            pos_emb.backward(d_enc, lr);
            token_emb.backward(d_enc, lr);
            
            return d_enc;
        }

        // Pooled path: input is [batch, n_elem, sub_toks] of token ids. Each element's sub-tokens
        // are embedded and mean-pooled into one vector, then run through pos_emb + blocks.
        // Output: [batch, n_elem, dim].
        Tensor forward_pooled(const Tensor& X)
        {
            int batch = X.shape[0];
            int n_elem = X.shape[1];
            int gsize = X.shape[2];

            Tensor flat_tokens = X.reshape({batch, n_elem * gsize});
            cached_pool_tokens = flat_tokens;
            Tensor emb = token_emb.forward(flat_tokens);   // [batch, n_elem*gsize, dim]

            int num_groups = batch * n_elem;
            Tensor pooled(std::vector<int>{num_groups, dim});
            // pad_id = 0 ([PAD] is special-token 0); pool only over real sub-tokens.
            mean_pool_groups_cuda(emb.data(), flat_tokens.data(), pooled.data(), num_groups, gsize, dim, 0.0f);
            cached_pool_groups = num_groups;
            cached_pool_gsize = gsize;

            Tensor x = pooled.reshape({batch, n_elem, dim});
            x = emb_drop.forward(x);
            
            // WE MUST NOT APPLY TRANSFORMER BLOCKS HERE! 
            // Applying self-attention across the 139 unordered schema elements mixes their vectors 
            // and completely destroys their individual identities, confusing the pointer network!
            
            return x;
        }

        Tensor backward_pooled(const Tensor& dY, float lr)
        {
            Tensor d = dY;
            d = emb_drop.backward(d, lr);

            // Un-pool: broadcast each element's gradient back across its real (non-pad) sub-tokens.
            int num_groups = cached_pool_groups;
            int gsize = cached_pool_gsize;
            Tensor d_unpooled(std::vector<int>{num_groups * gsize, dim});
            mean_pool_groups_backward_cuda(d.data(), cached_pool_tokens.data(), d_unpooled.data(), num_groups, gsize, dim, 0.0f);
            token_emb.backward(d_unpooled, lr);
            return Tensor();
        }

        void set_mode(bool train) override
        {
            token_emb.set_mode(train);
            pos_emb.set_mode(train);
            emb_drop.set_mode(train);
            for(auto& b : blocks) b->set_mode(train);
        }

        void save(const std::string& path)
        {
            std::ofstream os(path, std::ios::binary);
            if(!os.is_open()) throw std::runtime_error("Could not open file for saving: " + path);
            token_emb.save(os);
            pos_emb.save(os);
            for(auto& b : blocks) b->save(os);
            os.close();
            std::cout << "Saved TextEncoder to " << path << std::endl;
        }

        void load(const std::string& path)
        {
            std::ifstream is(path, std::ios::binary);
            if(!is.is_open()) throw std::runtime_error("Could not open file for loading: " + path);
            token_emb.load(is);
            pos_emb.load(is);
            for(auto& b : blocks) b->load(is);
            is.close();
            std::cout << "Loaded TextEncoder from " << path << std::endl;
        }
};
