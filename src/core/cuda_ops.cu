#include "../../include/core/tensor.h"
#include "../../include/core/context.h"
#include <cublas_v2.h>
#include <stdexcept>
#include <curand_kernel.h>
#include <cmath>

#define BLOCK_SIZE 256

/* 
Matrix multiplication kernel using SGEMM
SGEMM does column major multiplication, transpose input if needed.
C=alpha*op(A)*op(B)+beta*C, where op(X) is X or X^T depending on the transX flag.
Calculating C=AB is same as C^T=B^TA^T
*/
Tensor matrix_multiply(const Tensor& A,bool transA,const Tensor& B,bool transB)
{
    int m=transA?A.cols():A.rows();
    int k1=transA?A.rows():A.cols();
    int k2=transB?B.cols():B.rows();
    int n=transB?B.rows():B.cols();
    if(k1!=k2) throw std::invalid_argument("Inner dimensions must match for matrix multiplication");

    Tensor C(m,n);
    cublasHandle_t handle=CudaContext::get_instance().get_cublas_handle();

    cublasOperation_t opA=transA?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t opB=transB?CUBLAS_OP_T:CUBLAS_OP_N;

    float alpha=1.0f;
    float beta=0.0f;

    cublasStatus_t status=cublasSgemm(handle,opB,opA,n,m,k1,&alpha,B.data(),B.cols(),A.data(),A.cols(),&beta,C.data(),C.cols());
    if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS SGEMM failed!");
    return C;
}

/*
Image2Col and Col2Image kernels
im2col converts 3D image to 2D matrix, where each column is flattened kernel of image. col2im does reverse. These are used to avoid 3 nested loops in forward and backward pass.
im2col Mi,j=Xf(i,j) is one-one. Flattening 3D map onto 2D plane.
col2im dXf(i,j)+=dMi,j. Adding up gradients for each pixel in the image, since one pixel can be part of multiple kernels. Its many-one, thus need for atomic add.
*/

__global__ void im2col_kernel(const float* im,int n,int h,int w,int cin,int k_h,int k_w,int s,int p,int h_out,int w_out,float* m)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int cout=cin*k_h*k_w;
    int total=n*h_out*w_out*cout;
    if(index<total)
    {
        int c=index%cout;
        int r=index/cout;

        int w_out_idx=r%w_out;
        int h_out_idx=(r/w_out)%h_out;
        int n_idx=r/(w_out*h_out);

        int k_w_idx=c%k_w;
        int k_h_idx=(c/k_w)%k_h;
        int cin_idx=c/(k_h*k_w);

        int h_idx=h_out_idx*s+k_h_idx-p;
        int w_idx=w_out_idx*s+k_w_idx-p;

        if(h_idx>=0 && h_idx<h && w_idx>=0 && w_idx<w)
        {
            int im_idx=((n_idx*h+h_idx)*w+w_idx)*cin+cin_idx;
            m[index]=im[im_idx];
        }
        else m[index]=0.0f;
    }
}

__global__ void col2im_kernel(const float* m,int n,int h,int w,int cin,int k_h,int k_w,int s,int p,int h_out,int w_out,float* im)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int cout=cin*k_h*k_w;
    int total=n*h_out*w_out*cout;
    if(index<total)
    {
        int c=index%cout;
        int r=index/cout;

        int w_out_idx=r%w_out;
        int h_out_idx=(r/w_out)%h_out;
        int n_idx=r/(w_out*h_out);

        int k_w_idx=c%k_w;
        int k_h_idx=(c/k_w)%k_h;
        int cin_idx=c/(k_h*k_w);

        int h_idx=h_out_idx*s+k_h_idx-p;
        int w_idx=w_out_idx*s+k_w_idx-p;

        if(h_idx>=0 && h_idx<h && w_idx>=0 && w_idx<w)
        {
            int im_idx=((n_idx*h+h_idx)*w+w_idx)*cin+cin_idx;
            atomicAdd(&im[im_idx],m[index]);
        }
    }
}

void im2col(const Tensor& im,int k_h,int k_w,int s,int p,int h_out,int w_out,Tensor& m)
{
    int total=im.shape[0]*h_out*w_out*im.shape[3]*k_h*k_w;
    int threads=BLOCK_SIZE;
    int blocks=(total+threads-1)/threads;

    im2col_kernel<<<blocks,threads>>>(im.data(),im.shape[0],im.shape[1],im.shape[2],im.shape[3],k_h,k_w,s,p,h_out,w_out,m.data());
}

void col2im(const Tensor& m,int k_h,int k_w,int s,int p,int h_out,int w_out,Tensor& im)
{
    int total=im.shape[0]*h_out*w_out*im.shape[3]*k_h*k_w;
    int threads=BLOCK_SIZE;
    int blocks=(total+threads-1)/threads;
    cudaMemset(im.data(),0,im.rows()*im.cols()*sizeof(float));
    col2im_kernel<<<blocks,threads>>>(m.data(),im.shape[0],im.shape[1],im.shape[2],im.shape[3],k_h,k_w,s,p,h_out,w_out,im.data());
}