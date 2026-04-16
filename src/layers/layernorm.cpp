#include "../../include/layers/layernorm.h"
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f);
void layernorm_forward_cuda(const float* X,const float* g,const float* b,float* Y,float* mean,float* var,float* x_hat,int rows,int cols,float eps);
void layernorm_backward_cuda(const float* dY,const float* x_hat,const float* g,const float* var,float* dX,float* dg,float* db,int rows,int cols,float eps);

LayerNorm::LayerNorm(int dimension):dimension(dimension),t(0)
{
    std::vector<float> ones(dimension,1.0f);
    std::vector<float> zeros(dimension,0.0f);

    g=Tensor::upload(ones,1,dimension);
    b=Tensor::upload(zeros,1,dimension);

    dg=Tensor::zeros(1,dimension);
    db=Tensor::zeros(1,dimension);

    mg=Tensor::zeros(1,dimension); vg=Tensor::zeros(1,dimension);
    mb=Tensor::zeros(1,dimension); vb=Tensor::zeros(1,dimension);
}

Tensor LayerNorm::forward(const Tensor& input)
{
    cached_input=input;

    int cols=input.shape.back();
    int rows=(int)(input.total_elements()/cols);
    float eps=1e-5f;

    Tensor output(rows,cols);
    cached_mean=Tensor(1,rows);
    cached_var=Tensor(1,rows);
    cached_x=Tensor(rows,cols);

    layernorm_forward_cuda(input.data(),g.data(),b.data(),output.data(),cached_mean.data(),cached_var.data(),cached_x.data(),rows,cols,eps);

    if(input.shape.size()==3) output.shape=input.shape;

    return output;
}

Tensor LayerNorm::backward(const Tensor& dY,float lr)
{
    t++;

    int cols=cached_input.shape.back();
    int rows=(int)(cached_input.total_elements()/cols);
    float eps=1e-5f;

    Tensor dY_flat=dY.reshape({rows,cols});

    Tensor dX(rows,cols);
    dg.zero_();
    db.zero_();

    layernorm_backward_cuda(dY_flat.data(),cached_x.data(),g.data(),cached_var.data(),dX.data(),dg.data(),db.data(),rows,cols,eps);

    adam_cuda(g,dg,mg,vg,lr,t,dimension,0.0f);
    adam_cuda(b,db,mb,vb,lr,t,dimension,0.0f);

    if(cached_input.shape.size()==3) dX.shape=cached_input.shape;

    return dX;
}
