#include "../../include/layers/softmax.h"

void softmax_cuda(const Tensor& X,Tensor& Y);

Tensor Softmax::forward(const Tensor& input)
{
    Tensor output(input.rows(), input.cols());
    softmax_cuda(input,output);
    return output;
}

Tensor Softmax::backward(const Tensor& grad,float learning_rate)
{
    return grad;
}

