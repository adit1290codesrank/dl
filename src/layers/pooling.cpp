#include "../../include/layers/pooling.h"
#include <stdexcept>
#include <cuda_runtime.h>

void maxpool_forward_cuda(const float* X,float* Y,int* mask, int batch, int in_h, int in_w, int c, int size, int s, int out_h, int out_w);
void maxpool_backward_cuda(const float* dY,float* dX,const int* mask,int total_dy,int total_dx);

Pooling::Pooling(int size,int s):size(size),s(s),index(nullptr),cached_input_size(0),is_training(true){};

Pooling::~Pooling()
{
    if(index) cudaFree(index);
}

void Pooling::set_mode(bool training)
{
    this->is_training=training;
}

Tensor Pooling::forward(const Tensor& input)
{
    int n=input.shape[0];
    int h=input.shape[1];
    int w=input.shape[2];
    int c=input.shape[3];

    int OH=(h-size)/s+1;
    int OW=(w-size)/s+1;

    Tensor output({n,OH,OW,c});

    if(this->is_training)
    {
        if(mask==nullptr||(int)input.total_elements()!=this->cached_input_size)
        {
            if(mask) cudaFree(mask);
            cudaMalloc(&mask,n*OH*OW*c*sizeof(int));
        }
    }
    else
    {
        if(mask)
        {
            cudaFree(mask);
            mask=nullptr;
        }
    }

    this->cached_input_shape=input.shape;
    this->cached_input_size=(int)input.total_elements();
    
    maxpool_forward_cuda(input.data(),output.data(),mask,n,h,w,c,size,s,OH,OW);
    return output;
}

Tensor Pooling::backward(const Tensor& dY,float learning_rate)
{
    if(!this->is_training||mask==nullptr) return Tensor();
    Tensor dX(cached_input_shape);
    cudaMemset(dX.data(),0,dX.total_elements()*sizeof(float));

    maxpool_backward_cuda(dY.data(),dX.data(),mask,(int)dY.total_elements(),cached_input_size);
    return dX;
}

