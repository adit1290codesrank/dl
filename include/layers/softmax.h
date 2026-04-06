#pragma once
#include "layer.h"
#include "../include/core/tensor.h"

class Softmax:public Layer
{
    public:
        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad,float learning_rate) override;
};