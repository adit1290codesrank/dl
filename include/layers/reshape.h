#pragma once
#include "layer.h"
#include <vector>

class Reshape : public Layer 
{
private:
    std::vector<int> dim;      
    std::vector<int> cached_input_shape; 

public:
    Reshape(std::vector<int> shape):dim(shape) {}

    Tensor forward(const Tensor& input) override 
    {
        this->cached_input_shape=input.shape;

        std::vector<int> new_shape={input.shape[0]};
        new_shape.insert(new_shape.end(),dim.begin(),dim.end());

        return input.reshape(new_shape);
    }

    Tensor backward(const Tensor& dZ, float lr) override 
    {
        return dZ.reshape(cached_input_shape);
    }
};