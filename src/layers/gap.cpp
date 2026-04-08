#include "../../include/layers/gap.h"

void gap_forward_cuda(const float* input, float* output, int n, int c, int h, int w);
void gap_backward_cuda(const float* grad_output, float* grad_input, int n, int c, int h, int w);

Tensor GlobalAvgPool::forward(const Tensor& input) 
{
    this->input_shape = input.shape; 
    int n = input.rows();
    int c = input.shape[1];
    int h = input.shape[2];
    int w = input.shape[3];

    Tensor output = Tensor::zeros(n, c);
    
    gap_forward_cuda(input.data(), output.data(), n, c, h, w);
    return output;
}

Tensor GlobalAvgPool::backward(const Tensor& grad_output, float lr)
{
    int n = input_shape[0];
    int c = input_shape[1];
    int h = input_shape[2];
    int w = input_shape[3];

    Tensor grad_input = Tensor::zeros(n, c * h * w);
    
    gap_backward_cuda(grad_output.data(), grad_input.data(), n, c, h, w);
    return grad_input;
}