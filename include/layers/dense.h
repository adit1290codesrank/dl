#pragma once
#include <vector>
#include <random>
#include "layer.h"
#include "../core/tensor.h"

class Dense:public Layer
{
    private:
        int input_size;
        int output_size;

        Tensor w;
        Tensor b;

        Tensor mw,vw;
        Tensor mb,vb;
        int t;

    public:
        Dense(int input_size,int output_size);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad,float learning_rate) override;

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
};