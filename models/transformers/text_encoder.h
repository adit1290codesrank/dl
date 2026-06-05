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

    public:
        TextEncoder(int vocab_size, int seq_len, int dim, int heads, int depth)
            : vocab_size(vocab_size), seq_len(seq_len), dim(dim), depth(depth),
              token_emb(vocab_size, dim), pos_emb(seq_len, dim), emb_drop(0.4f)
        {
            for(int i = 0; i < depth; ++i) 
                blocks.push_back(std::make_unique<Transformer>(dim, heads, false));
        }

        Tensor forward(const Tensor& X) override
        {
            int batch = X.shape[0];

            Tensor x = token_emb.forward(X);

            std::vector<float> pidx(batch * seq_len);
            for(int b = 0; b < batch; ++b) {
                for(int t = 0; t < seq_len; ++t) {
                    pidx[b * seq_len + t] = (float)t;
                }
            }
            
            x = x + pos_emb.forward(Tensor::upload(pidx, {batch, seq_len}));
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
