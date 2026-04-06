#include "../../include/layers/activation.h"
#include <stdexcept>

void leaky_relu_forward_cuda(Tensor& Y,float alpha);
void leaky_relu_backward_cuda(const Tensor& dY,const Tensor& cached_X,Tensor& dX,float alpha);

Activation::Activation(ActivationType t,float alpha):type(t),alpha(alpha){}

Tensor Activation::forward(const Tensor& input)
{
    this->cached_input=input;
    
    Tensor Y=input.clone();
    if(type==ActivationType::LEAKY_RELU) leaky_relu_forward_cuda(Y,this->alpha);
    else throw std::runtime_error("Unsupported activation type");
    return Y;
}

Tensor Activation::backward(const Tensor& grad,float learning_rate)
{
    Tensor dX(grad.shape);
    if(type==ActivationType::LEAKY_RELU) leaky_relu_backward_cuda(grad,this->cached_input,dX,this->alpha);
    else throw std::runtime_error("Unsupported activation type");
    return dX;
}