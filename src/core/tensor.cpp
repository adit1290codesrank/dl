#include "../../include/core/tensor.h"
#include <cuda_runtime.h>
#include <numeric>
#include <stdexcept>

Tensor matrix_multiply(const Tensor& A,bool transA,const Tensor& B,bool transB);
void im2col_cuda(const Tensor& im,int k_h,int k_w,int s,int p,int h_out,int w_out,Tensor& m);
void col2im_cuda(const Tensor& m,int k_h,int k_w,int s,int p,int h_out,int w_out,Tensor& im);

Tensor::Tensor(int r,int c)
{
    this->shape={r,c};
    float *ptr=nullptr;
    if(cudaMalloc(&ptr,r*c*sizeof(float))!=cudaSuccess) throw std::runtime_error("Failed to allocate memory on GPU");
    d_ptr=std::shared_ptr<float>(ptr,[](float* p){cudaFree(p);});
}

Tensor::Tensor(std::vector<int> shape)
{
    this->shape=shape;
    int total=1;
    for(int dim:shape) total*=dim;

    float *ptr=nullptr;
    if(cudaMalloc(&ptr,total*sizeof(float))!=cudaSuccess) throw std::runtime_error("Failed to allocate memory on GPU");
    this->d_ptr=std::shared_ptr<float>(ptr,[](float* p){cudaFree(p);});
}

float* Tensor::data() const {return d_ptr.get();}

Tensor Tensor::upload(const std::vector<float>& upload_data,int r,int c)
{
    if(upload_data.size()!=r*c) throw std::invalid_argument("Data size does not match specified shape");
    Tensor t(r,c);
    cudaMemcpy(t.data(),upload_data.data(),upload_data.size()*sizeof(float),cudaMemcpyHostToDevice);
    return t;
}

Tensor Tensor::upload(const std::vector<float>& upload_data,const std::vector<int>& shape)
{
    Tensor t(shape);
    cudaMemcpy(t.data(),upload_data.data(),upload_data.size()*sizeof(float),cudaMemcpyHostToDevice);
    return t;
}

std::vector<float> Tensor::download() const
{
    std::vector<float> download_data(this->rows()*this->cols());
    cudaMemcpy(download_data.data(),this->data(),download_data.size()*sizeof(float),cudaMemcpyDeviceToHost);
    return download_data;
}

Tensor Tensor::zeros(int r,int c)
{
    Tensor t(r,c);
    cudaMemset(t.data(),0,r*c*sizeof(float));
    return t;
}

Tensor Tensor::slice(int start_row,int num_rows) const
{
    if(start_row<0 || start_row+num_rows>this->rows()) throw std::out_of_range("Slice indices out of range");

    Tensor t(num_rows,this->cols());
    if(!this->shape.empty())
    {
        t.shape=this->shape;
        t.shape[0]=num_rows;
    }
    else t.shape={num_rows,this->cols()};
    
    float *ptr=this->data()+(start_row*this->cols());
    t.d_ptr=std::shared_ptr<float>(this->d_ptr,ptr);
    return t;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    return matrix_multiply(*this,false,other,false);
}

size_t Tensor::total_elements() const
{
    if(shape.empty()) return 0;
    size_t total=1;
    for(int dim:shape) total*=dim;
    return total;
}

Tensor Tensor::reshape(std::vector<int> shape) const
{
    size_t total=1;
    for(int dim:shape) total*=dim;
    if(total!=this->total_elements()) throw std::invalid_argument("Total elements must remain the same in reshape");

    Tensor t=*this;
    t.shape=shape;
    return t;
}

Tensor Tensor::flatten() const
{
    return this->reshape({this->shape[0],(int)(this->total_elements()/this->shape[0])});
}

Tensor Tensor::im2col(int k_h,int k_w,int s,int p) const
{
    int n=shape[0],h=shape[1],w=shape[2],c=shape[3];
    int h_out=(h+2*p-k_h)/s+1;
    int w_out=(w+2*p-k_w)/s+1;

    Tensor m(n*h_out*w_out,c*k_h*k_w);
    im2col_cuda(*this,k_h,k_w,s,p,h_out,w_out,m);
    return m;
}

Tensor Tensor::col2im(const std::vector<int>& shape,int k_h,int k_w,int s,int p) const
{
    Tensor im(shape);
    int h_out=(shape[1]+2*p-k_h)/s+1;
    int w_out=(shape[2]+2*p-k_w)/s+1;
    col2im_cuda(*this,k_h,k_w,s,p,h_out,w_out,im);
    return im;
}







