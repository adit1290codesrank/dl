#include "../../include/layers/embedding.h"
#include <stdexcept>
#include <random>

void embedding_forward_cuda(const float* X,const float* W,float* Y,int tokens,int dimension,int size);
void embedding_backward_cuda(const float* X,const float* dY,float* dW,int tokens,int dimension,int size);
void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f);

Embedding::Embedding(int size,int dimension):size(size),dimension(dimension)
{
    this->t=0;
    std::vector<float> h_w(size*dimension);
    std::mt19937 gen(42);

    float stddev=sqrt(2.0f/dimension);
    std::normal_distribution<float> dist(0.0f,stddev);

    for(int i=0;i<size*dimension;i++) h_w[i]=dist(gen);

    this->w=Tensor::upload(h_w,size,dimension);
    this->dw=Tensor::zeros(size,dimension);

    this->mw=Tensor::zeros(size,dimension);
    this->vw=Tensor::zeros(size,dimension);
}

Tensor Embedding::forward(const Tensor& input)
{
    this->cached_input=input;
    std::vector<int> shape=input.shape;
    shape.push_back(dimension);
    Tensor output(shape);
    embedding_forward_cuda(input.data(),w.data(),output.data(),input.total_elements(),dimension,size);
    return output;
}

Tensor Embedding::backward(const Tensor& dY, float lr)
{
    dw.zero_();
    embedding_backward_cuda(cached_input.data(),dY.data(),dw.data(),cached_input.total_elements(),dimension,size);
    
    t++;
    adam_cuda(w,dw,mw,vw,lr,t,w.total_elements());

    return Tensor();
}