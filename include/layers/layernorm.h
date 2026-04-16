#pragma once
#include "layer.h"
#include "../core/tensor.h"

class LayerNorm:public Layer
{
    private:
        int dimension;
        int t;
        
        Tensor g,b;
        Tensor dg,db;

        Tensor mg,vg;
        Tensor mb,vb;

        Tensor cached_input;
        Tensor cached_mean,cached_var,cached_x;

    public:
        LayerNorm(int dimension);
        ~LayerNorm()=default;

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& dY, float lr) override;
};