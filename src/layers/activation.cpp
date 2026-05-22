#include "../../include/layers/activation.h"
#include <stdexcept>

void leaky_relu_forward_cuda(Tensor& Y,float alpha);
void leaky_relu_backward_cuda(const Tensor& dY,const Tensor& cached_X,Tensor& dX,float alpha);
void sigmoid_forward_cuda(Tensor& Y);
void sigmoid_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX);
void gelu_forward_cuda(Tensor& Y);
void gelu_backward_cuda(const Tensor& dY,const Tensor& cached_X,Tensor& dX);
void tanh_forward_cuda(Tensor& Y);
void tanh_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX);
void relu_forward_cuda(Tensor& Y);
void relu_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX);
void silu_forward_cuda(Tensor& Y);
void silu_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX);

Activation::Activation(ActivationType t,float alpha):type(t),alpha(alpha){}

Tensor Activation::forward(const Tensor& input)
{
    this->cached_input=input;
    
    Tensor Y=input.clone();
    if(type==ActivationType::LEAKY_RELU) leaky_relu_forward_cuda(Y,this->alpha);
    else if(type==ActivationType::SIGMOID) sigmoid_forward_cuda(Y);
    else if(type==ActivationType::GELU) gelu_forward_cuda(Y);
    else if(type==ActivationType::TANH) tanh_forward_cuda(Y);
    else if(type==ActivationType::RELU) relu_forward_cuda(Y);
    else if(type==ActivationType::SILU) silu_forward_cuda(Y);
    else throw std::runtime_error("Unsupported activation type");
    return Y;
}

Tensor Activation::backward(const Tensor& grad,float learning_rate)
{
    Tensor dX(grad.shape);
    if(type==ActivationType::LEAKY_RELU) leaky_relu_backward_cuda(grad,this->cached_input,dX,this->alpha);
    else if(type==ActivationType::SIGMOID) sigmoid_backward_cuda(grad,this->cached_input,dX);
    else if(type==ActivationType::GELU) gelu_backward_cuda(grad,this->cached_input,dX);
    else if(type==ActivationType::TANH) tanh_backward_cuda(grad,this->cached_input,dX);
    else if(type==ActivationType::RELU) relu_backward_cuda(grad,this->cached_input,dX);
    else if(type==ActivationType::SILU) silu_backward_cuda(grad,this->cached_input,dX);
    else throw std::runtime_error("Unsupported activation type");
    return dX;
}