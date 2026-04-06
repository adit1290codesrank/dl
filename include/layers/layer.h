#pragma once
#include "../core/tensor.h"

//Parent class for all layers
class Layer
{
    protected:
        Tensor cached_input;

    public:
        virtual ~Layer()=default;

        virtual Tensor forward(const Tensor& input)=0;//Forward pass,Y=WX+B and Z=activation(Y)
        virtual Tensor backward(const Tensor& grad,float learning_rate)=0;//Backward pass,grad is dL/dZ,returns dL/dX

        virtual void save(std::ofstream& os) {}
        virtual void load(std::ifstream& is) {}

        virtual void set_mode(bool training) {}
};

