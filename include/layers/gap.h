#pragma once
#include "layer.h"

class GlobalAvgPool:public Layer 
{
public:
    GlobalAvgPool()=default;
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, float lr) override;
    
private:
    std::vector<int> input_shape;
};