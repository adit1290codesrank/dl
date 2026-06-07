#pragma once
#include "layer.h"
#include <fstream>
#include "../core/tensor.h"

class Embedding:public Layer
{
    private:
        int size;
        int dimension;

        Tensor w,dw;

        int t;
        Tensor mw,vw;

        Tensor cached_input;

        // Weight tying: an output projection sharing this matrix stashes its grad here,
        // to be folded into the single Adam step alongside the lookup gradient.
        Tensor pending_ext_grad;
        bool has_ext = false;

    public:
        Embedding(int size,int dimension);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad, float lr) override;

        // Tied-weight access. weight() is [size, dimension] (= [vocab, dim] for token_emb).
        Tensor& weight() { return w; }
        int emb_size() const { return size; }
        int emb_dim() const { return dimension; }
        void add_external_grad(const Tensor& g);

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
};

