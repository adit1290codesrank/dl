#pragma once
#include "layer.h"
#include <fstream>
#include "../core/tensor.h"
#include "self_attention.h"
#include "layernorm.h"
#include "dense.h"
#include "activation.h"

class Transformer:public Layer
{
    private:
        int dimension;
        
        SelfAttention attention;
        LayerNorm ln1,ln2;

        Dense d1,d2;
        Activation act;

        Tensor cached_input;
        Tensor cached_output1;

    public:
        Transformer(int dimension, int heads, bool causal=true);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& dY, float lr) override;

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
        void set_mode(bool train) override;
};