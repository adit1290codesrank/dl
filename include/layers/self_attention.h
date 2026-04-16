#pragma once
#include "layer.h"
#include "../core/tensor.h"

class SelfAttention:public Layer
{
    private:
        int dimension;
        int heads;
        float scale;

        Tensor wQ,wK,wV,wO;
        Tensor dwQ,dwK,dwV,dwO;

        int t;
        Tensor mwq,vwq;
        Tensor mwk,vwk;
        Tensor mwv,vwv;
        Tensor mwo,vwo;

        Tensor cached_input;
        Tensor cachedQ,cachedK,cachedV;
        Tensor cached_attention;

    public:
        SelfAttention(int dimension,int heads);
        
        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad, float lr) override;
};