#pragma once
#include "layer.h"
#include "../core/tensor.h"

enum class ActivationType
{
    LEAKY_RELU
};

class Activation:public Layer
{
    private:
        ActivationType type;
        float alpha;
    
    public:
        Activation(ActivationType t=ActivationType::LEAKY_RELU,float alpha=0.01f);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad,float learning_rate) override;
};
