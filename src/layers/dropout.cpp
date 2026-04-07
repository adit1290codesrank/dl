#include "../../include/layers/dropout.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <chrono>

void dropout_forward_cuda(const float* X,float* Y,float* mask,int size,float rate,unsigned long long seed);
void dropout_backward_cuda(const float* dY,float* dX,const float* mask,int size);

Dropout::Dropout(float rate):rate(rate),mask(nullptr),is_training(true)
{
    auto now=std::chrono::high_resolution_clock::now();
    this->seed=static_cast<unsigned long long>(now.time_since_epoch().count())^reinterpret_cast<std::uintptr_t>(this);
}

Dropout::~Dropout()
{
    if(mask) cudaFree(mask);
}

void Dropout::set_mode(bool training)
{
    this->is_training=training;
}

Tensor Dropout::forward(const Tensor& X)
{
    if(!is_training) return X;

    int size=X.total_elements();
    Tensor Y(X.shape);

    if(mask) cudaFree(mask);
    cudaMalloc(&mask,size*sizeof(float));

    dropout_forward_cuda(X.data(),Y.data(),mask,size,rate,seed++);
    return Y;
}

Tensor Dropout::backward(const Tensor& dY,float learning_rate)
{
    if(!is_training) return dY;

    Tensor dX(dY.shape);
    dropout_backward_cuda(dY.data(),dX.data(),mask,dY.total_elements());
    return dX;
}