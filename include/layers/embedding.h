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

    public:
        Embedding(int size,int dimension);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad, float lr) override;

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
};

