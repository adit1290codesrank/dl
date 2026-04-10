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

Tensor matrix_add(const Tensor& A,const Tensor& B)
{
    if(A.shape != B.shape) throw std::invalid_argument("Tensor shapes must match for matrix addition");

    Tensor C(A.shape);
    cublasHandle_t handle=CudaContext::get_instance().get_cublas_handle();

    float alpha=1.0f;
    float beta=1.0f;

    int total=1;
    for(auto dim:A.shape) total*=dim;
    cublasStatus_t status=cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,total,1,&alpha,A.data(),total,&beta,B.data(),total,C.data(),total);

    if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS SGEAM failed!");
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



/*
MSE Kernel.
Error=1/N*sum((y_-y)^2)
dError/dy=2/N*(y_-y)
*/
__global__ void mse_kernel(const float* y_,const float* y, float* dy,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) dy[index]=2.0f*(y_[index]-y[index])/(float)size;
}

__global__ void mse_loss_kernel(const float* y_,const float* y, float* loss,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) atomicAdd(loss, (y_[index]-y[index])*(y_[index]-y[index])/(float)size);
}

/*
Cross Entropy Kernel.
Error=-1/N*sum(y_*log(y))
dError/dy=-1/N*(y_/y)
*/
__global__ void cross_entropy_kernel(const float* y_, const float* y, float* dy,int size,int n,int k,float epsilon)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size)
    {
        float target=y[index]*(1.0f-epsilon)+(epsilon/(float)k);
        dy[index]=(y_[index]-target)/(float)n;
    }
}

__global__ void cross_entropy_loss_kernel(const float* y_, const float* y,float* loss, int size, int n,int k, float epsilon)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        float target = y[index] * (1.0f - epsilon) + (epsilon / (float)k);
        atomicAdd(loss, -target * logf(y_[index] + 1e-7f) / (float)n);
    }
}

__global__ void add_bias_kernel(float* Y,float* b,int n,int m)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int total=n*m;
    if(index<total) Y[index]+=b[index%m];
}

__global__ void sum_rows_kernel(const float* dY,float* db,int n,int m)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<m)
    {
        float sum=0.0f;
        for(int i=0;i<n;i++) sum+=dY[i*m+index];
        db[index]=sum;
    }
}

__global__ void adam_kernel(float* w,const float* grad,float* m,float* v,float lr,int t,int size,float lambda) 
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size) 
    {
        float g=grad[idx]+(lambda*w[idx]);
        
        float mt=0.9f*m[idx]+(1.0f-0.9f)*g;
        float vt=0.999f*v[idx]+(1.0f-0.999f)*g*g;
        
        m[idx]=mt;
        v[idx]=vt;
        
        mt=mt/(1.0f-powf(0.9f,t));
        vt=vt/(1.0f-powf(0.999f, t));
        
        w[idx]-=lr*mt/(sqrtf(vt)+1e-8f);
    }
}

__global__ void leaky_relu_forward_kernel(float* Y,float alpha,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) Y[index]=Y[index]>0.0f?Y[index]:alpha*Y[index];
}

__global__ void leaky_relu_backward_kernel(const float* dY,const float* cached_X,float* dX,float alpha,int size)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) dX[index]=cached_X[index]>0.0f?dY[index]:alpha*dY[index];
}

__global__ void softmax_kernel(const float* X,float* Y,int n,int m)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<n)
    {
        int row_start=index*m;
        float max_val=X[row_start];
        for(int i=1;i<m;i++) if(X[row_start+i]>max_val) max_val=X[row_start+i];

        float sum=0.0f;
        for(int i=0;i<m;i++)
        {
            Y[row_start+i]=expf(X[row_start+i]-max_val);
            sum+=Y[row_start+i];
        }
        for(int i=0;i<m;i++) Y[row_start+i]/=sum;
    }
}

__global__ void bn_backward_gamma_beta_kernel(const float* dY, const float* x_hat, float* d_gamma, float* d_beta, int N, int C) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (col < C) {
        float sum_dgamma = 0.0f;
        float sum_dbeta = 0.0f;
        
        for (int row = 0; row < N; row++) {
            int idx = row * C + col;
            sum_dgamma += dY[idx] * x_hat[idx];
            sum_dbeta += dY[idx];
        }
        
        d_gamma[col] = sum_dgamma;
        d_beta[col] = sum_dbeta;
    }
}


__global__ void bn_backward_x_kernel(const float* dY, const float* x_hat, const float* var, const float* gamma, const float* d_gamma, const float* d_beta, float* dX, float eps, int N, int C) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C;

    if (idx < total_elements) {
        int col = idx % C; 
        
        float inv_std = 1.0f / sqrtf(var[col] + eps);
        float norm_dgamma = d_gamma[col] / N;
        float norm_dbeta = d_beta[col] / N;
        
        dX[idx] = gamma[col] * inv_std * (dY[idx] - norm_dbeta - x_hat[idx] * norm_dgamma);
    }
}

__global__ void compute_batch_stats_kernel(const float* x, float* mean, float* var, int N, int C) 
{
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sq_sum[BLOCK_SIZE];

    int col = blockIdx.x; 
    int tid = threadIdx.x;

    if (col >= C) return;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int row = tid; row < N; row += blockDim.x) 
    {
        float val = x[row * C + col];
        local_sum += val;
        local_sq_sum += val * val;
    }

    s_sum[tid] = local_sum;
    s_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) 
    {
        float m = s_sum[0] / N;
        mean[col] = m;
        float v = (s_sq_sum[0] / N) - (m * m);
        var[col] = (v < 0.0f) ? 0.0f : v; 
    }
}

__global__ void apply_batchnorm_kernel(const float* x, const float* mean, const float* var, const float* gamma, const float* beta, float* x_hat, float* out, float eps, int N, int C) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    int total_elements = N * C;

    if (idx < total_elements) {
        int col = idx % C; 

        float normalized = (x[idx] - mean[col]) / sqrtf(var[col] + eps);
        x_hat[idx] = normalized;

        out[idx] = gamma[col] * normalized + beta[col];
    }
}

__global__ void update_running_stats_kernel(float* r_mean, float* r_var, const float* b_mean, const float* b_var, float momentum, int C) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < C) {
        r_mean[col] = (1.0f - momentum) * r_mean[col] + momentum * b_mean[col];
        r_var[col] = (1.0f - momentum) * r_var[col] + momentum * b_var[col];
    }
}

__global__ void maxpool_forward_kernel(const float* X,float* Y,int* mask,int batch, int in_h,int in_w,int c,int size,int s,int out_h,int out_w)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<batch*out_h*out_w*c)
    {
        int col=index%c;
        int w_out=(index/c)%out_w;
        int h_out=(index/(c*out_w))%out_h;
        int b=index/(c*out_w*out_h);

        int h_start=h_out*s;
        int w_start=w_out*s;    
        
        float max_val=-1e38f;
        int max_idx=-1;

        for(int i=0;i<size;i++)
        {
            for(int j=0;j<size;j++)
            {
                int cur_h=h_start+i;
                int cur_w=w_start+j;
                if(cur_h<in_h && cur_w<in_w)
                {
                    int idx=((b*in_h+cur_h)*in_w+cur_w)*c+col;
                    if(X[idx]>max_val)
                    {
                        max_val=X[idx];
                        max_idx=idx;
                    }
                }
            }
        }
        Y[index]=max_val;
        if(mask!=nullptr) mask[index]=max_idx;
    }
}

__global__ void maxpool_backward_kernel(const float* dY,float* dX,const int* mask,int total) 
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<total) 
    {
        int target_idx=mask[index];
        atomicAdd(&dX[target_idx],dY[index]);
    }
}

__global__ void dropout_forward_kernel(const float* X,float* Y,float *mask,int size,float rate,float scale,unsigned long long seed)
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed,index,0,&state);

        float val=curand_uniform(&state);
        if(val<rate)
        {
            mask[index]=0.0f;
            Y[index]=0.0f;
        }
        else
        {
            mask[index]=scale;
            Y[index]=X[index]*scale;
        }
    }
}

__global__ void dropout_backward_kernel(const float* dY,float* dX,const float* mask,int size) 
{
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) dX[index]=dY[index]*mask[index];

}

__global__ void gap_forward_kernel(const float* input, float* output,int n, int c, int h, int w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < n * c)
    {
        int c_idx = idx % c;
        int n_idx = idx / c;
        float sum = 0.0f;
        for (int hi = 0; hi < h; hi++)
            for (int wi = 0; wi < w; wi++)
                sum += input[((n_idx * h + hi) * w + wi) * c + c_idx]; 
        output[idx] = sum / (float)(h * w);
    }
}

__global__ void gap_backward_kernel(const float* grad_output, float* grad_input,int n, int c, int h, int w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * h * w * c)
    {
        int c_idx = idx % c;
        int n_idx = (idx / c) / (h * w);
        grad_input[idx] = grad_output[n_idx * c + c_idx] / (float)(h * w);
    }
}

__global__ void gen_aug_params_kernel(int* tops, int* lefts, int* flips,int N, int pad, unsigned long long seed)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, n, 0, &state);

    int range = 2 * pad + 1;
    tops[n]  = min((int)(curand_uniform(&state) * range), 2 * pad);
    lefts[n] = min((int)(curand_uniform(&state) * range), 2 * pad);
    flips[n] = curand_uniform(&state) < 0.5f ? 1 : 0;
}

__global__ void apply_aug_kernel(const float* input, float* output,const int* tops, const int* lefts, const int* flips,int N, int H, int W, int C, int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * W * C) return;

    int c =  idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n =  idx / (C * W * H);

    int src_h = h + tops[n] - pad;
    int src_w = (flips[n] ? (W - 1 - w) : w) + lefts[n] - pad;

    output[idx] = (src_h >= 0 && src_h < H && src_w >= 0 && src_w < W)? input[((n * H + src_h) * W + src_w) * C + c]: 0.0f;
}

__global__ void aug_backward_kernel(const float* dout, float* din,const int* tops, const int* lefts, const int* flips,int N, int H, int W, int C, int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * W * C) return;

    int c =  idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n =  idx / (C * W * H);

    int src_h = h + tops[n] - pad;
    int src_w = (flips[n] ? (W - 1 - w) : w) + lefts[n] - pad;

    if (src_h >= 0 && src_h < H && src_w >= 0 && src_w < W)atomicAdd(&din[((n * H + src_h) * W + src_w) * C + c], dout[idx]);
}

__global__ void cutout_kernel(float* X, const int* cx, const int* cy,int N, int H, int W, int C, int cut_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * H * W * C) return;

    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n =  idx / (C * W * H);

    int half = cut_size / 2;
    int x0 = cx[n] - half, x1 = cx[n] + half;
    int y0 = cy[n] - half, y1 = cy[n] + half;

    if (h >= y0 && h < y1 && w >= x0 && w < x1) X[idx] = 0.0f;
}

__global__ void gen_cutout_params_kernel(int* cx, int* cy,int N, int H, int W,unsigned long long seed)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, n, 0, &state);

    cx[n] = (int)(curand_uniform(&state) * W);
    cy[n] = (int)(curand_uniform(&state) * H);
}

void cutout_cuda(Tensor& X, int* d_cx, int* d_cy,int cut_size, unsigned long long seed)
{
    int N = X.shape[0];
    int H = X.shape[1];
    int W = X.shape[2];
    int C = X.shape[3];

    int pblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gen_cutout_params_kernel<<<pblocks, BLOCK_SIZE>>>(d_cx, d_cy, N, H, W, seed);

    int total = N * H * W * C;
    int ablocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cutout_kernel<<<ablocks, BLOCK_SIZE>>>(X.data(), d_cx, d_cy, N, H, W, C, cut_size);
}

void augment_forward_cuda(const Tensor& input, Tensor& output,int* d_tops, int* d_lefts, int* d_flips,int pad, unsigned long long seed)
{
    int N = input.shape[0];
    int H = input.shape[1];
    int W = input.shape[2];
    int C = input.shape[3];

    int pblocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gen_aug_params_kernel<<<pblocks, BLOCK_SIZE>>>(d_tops, d_lefts, d_flips, N, pad, seed);

    int total = N * H * W * C;
    int ablocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_aug_kernel<<<ablocks, BLOCK_SIZE>>>(
        input.data(), output.data(), d_tops, d_lefts, d_flips, N, H, W, C, pad);
}

void augment_backward_cuda(const Tensor& dout, Tensor& din,const int* d_tops, const int* d_lefts, const int* d_flips,int pad)
{
    int N = dout.shape[0];
    int H = dout.shape[1];
    int W = dout.shape[2];
    int C = dout.shape[3];

    cudaMemset(din.data(), 0, din.total_elements() * sizeof(float));

    int total = N * H * W * C;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    aug_backward_kernel<<<blocks, BLOCK_SIZE>>>(dout.data(), din.data(), d_tops, d_lefts, d_flips, N, H, W, C, pad);
}

void gap_forward_cuda(const float* input, float* output, int n, int c, int h, int w)
{
    int total_nc = n * c;
    int blocks = (total_nc + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gap_forward_kernel<<<blocks, BLOCK_SIZE>>>(input, output, n, c, h, w);
}

void gap_backward_cuda(const float* grad_output, float* grad_input, int n, int c, int h, int w)
{
    int total = n * h * w * c;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gap_backward_kernel<<<blocks, BLOCK_SIZE>>>(grad_output, grad_input, n, c, h, w);
}

void dropout_forward_cuda(const float* X,float* Y,float* mask,int size,float rate,unsigned long long seed)
{
    int threads=BLOCK_SIZE;
    int blocks=(size+threads-1)/threads;
    float scale=1.0f/(1.0f-rate);
    dropout_forward_kernel<<<blocks,threads>>>(X,Y,mask,size,rate,scale,seed);
}

void dropout_backward_cuda(const float* dY,float* dX,const float* mask,int size) 
{
    int threads=256;
    int blocks=(size+threads-1)/threads;
    dropout_backward_kernel<<<blocks,threads>>>(dY,dX,mask,size);
}

void maxpool_forward_cuda(const float* X,float* Y,int* mask, int batch, int in_h, int in_w, int c, int size, int s, int out_h, int out_w)
{
    int total=batch*out_h*out_w*c;
    int threads=BLOCK_SIZE;
    int blocks=(total+threads-1)/threads;
    maxpool_forward_kernel<<<blocks,threads>>>(X,Y,mask,batch,in_h,in_w,c,size,s,out_h,out_w);
}

void maxpool_backward_cuda(const float* dY,float* dX,const int* mask,int total_dy,int total_dx) 
{
    cudaMemset(dX,0,total_dx*sizeof(float));
    int threads=BLOCK_SIZE;
    int blocks=(total_dy+threads-1)/threads;
    maxpool_backward_kernel<<<blocks,threads>>>(dY,dX,mask,total_dy);
}

void batchnorm_forward_cuda(const Tensor& dY, const Tensor& x_hat, Tensor& d_gamma, Tensor& d_beta) 
{
    int threads = 256;
    int blocks = (dY.cols() + threads - 1) / threads;
    bn_backward_gamma_beta_kernel<<<blocks, threads>>>(dY.data(), x_hat.data(), d_gamma.data(), d_beta.data(), dY.rows(), dY.cols());
}

void batchnorm_backward_cuda(const Tensor& dY, const Tensor& x_hat, const Tensor& var, const Tensor& gamma, const Tensor& d_gamma, const Tensor& d_beta, Tensor& dX, float epsilon) 
{
    int total_elements = dY.rows() * dY.cols();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    bn_backward_x_kernel<<<blocks, threads>>>(dY.data(), x_hat.data(), var.data(), gamma.data(), d_gamma.data(), d_beta.data(), dX.data(), epsilon, dY.rows(), dY.cols());
}

void compute_batch_stats_cuda(const Tensor& input, Tensor& mean, Tensor& var) 
{
    int N = input.rows(); 
    int C = input.cols(); 

    dim3 blocks(C);
    dim3 threads(BLOCK_SIZE);

    compute_batch_stats_kernel<<<blocks, threads>>>(input.data(), mean.data(), var.data(), N, C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel error in compute_batch_stats: %s\n", cudaGetErrorString(err));
    }
}

void batchnorm_cuda(const Tensor& x, const Tensor& mean, const Tensor& var, const Tensor& gamma, const Tensor& beta, Tensor& x_hat, Tensor& output, float epsilon) 
{
    int total_elements = x.rows() * x.cols();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    apply_batchnorm_kernel<<<blocks, threads>>>(x.data(), mean.data(), var.data(), gamma.data(), beta.data(), x_hat.data(), output.data(), epsilon, x.rows(), x.cols());
}

void update_running_stats_cuda(Tensor& r_mean, Tensor& r_var, const Tensor& b_mean, const Tensor& b_var, float momentum) 
{
    int threads = 256;
    int blocks = (r_mean.cols() + threads - 1) / threads;
    update_running_stats_kernel<<<blocks, threads>>>(r_mean.data(), r_var.data(), b_mean.data(), b_var.data(), momentum, r_mean.cols());
}

void softmax_cuda(const Tensor& X,Tensor& Y)
{
    int threads=BLOCK_SIZE;
    int blocks=(X.rows()+threads-1)/threads;
    softmax_kernel<<<blocks,threads>>>(X.data(),Y.data(),X.rows(),X.cols());
}

void leaky_relu_forward_cuda(Tensor& Y,float alpha)
{
    int size=Y.rows()*Y.cols();
    int threads=BLOCK_SIZE;
    int blocks=(size+threads-1)/threads;
    leaky_relu_forward_kernel<<<blocks,threads>>>(Y.data(),alpha,size);
}

void leaky_relu_backward_cuda(const Tensor& dY,const Tensor& cached_X,Tensor& dX,float alpha)
{
    int size=dY.rows()*dY.cols();
    int threads=BLOCK_SIZE;
    int blocks=(size+threads-1)/threads;
    leaky_relu_backward_kernel<<<blocks,threads>>>(dY.data(),cached_X.data(),dX.data(),alpha,size);
}

void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f)
{
    int threads=BLOCK_SIZE;
    int blocks=(size+threads-1)/threads;
    adam_kernel<<<blocks,threads>>>(W.data(),grad.data(),m.data(),v.data(),lr,t,size,lamda);
}

void sum_rows_cuda(const Tensor& dY, Tensor& db)
{
    int total=dY.cols();
    int threads=BLOCK_SIZE;
    int blocks=(total+threads-1)/threads;
    sum_rows_kernel<<<blocks,threads>>>(dY.data(),db.data(),dY.rows(),dY.cols());
}

void add_bias_cuda(Tensor& Y, const Tensor& b)
{
    int total=Y.total_elements();
    int threads=BLOCK_SIZE;
    int blocks=(total+threads-1)/threads;
    add_bias_kernel<<<blocks,threads>>>(Y.data(),b.data(),Y.rows(),Y.cols());
}

void im2col_cuda(const Tensor& im,int k_h,int k_w,int s,int p,int h_out,int w_out,Tensor& m)
{
    int total=im.shape[0]*h_out*w_out*im.shape[3]*k_h*k_w;
    int threads=BLOCK_SIZE;
    int blocks=(total+threads-1)/threads;

    im2col_kernel<<<blocks,threads>>>(im.data(),im.shape[0],im.shape[1],im.shape[2],im.shape[3],k_h,k_w,s,p,h_out,w_out,m.data());
}

void col2im_cuda(const Tensor& m,int k_h,int k_w,int s,int p,int h_out,int w_out,Tensor& im)
{
    int total=im.shape[0]*h_out*w_out*im.shape[3]*k_h*k_w;
    int threads=BLOCK_SIZE;
    int blocks=(total+threads-1)/threads;
    cudaMemset(im.data(),0,im.rows()*im.cols()*sizeof(float));
    col2im_kernel<<<blocks,threads>>>(m.data(),im.shape[0],im.shape[1],im.shape[2],im.shape[3],k_h,k_w,s,p,h_out,w_out,im.data());
}

void mse_cuda(const Tensor& y_,const Tensor& y,Tensor& dy)
{
    int size=y.rows()*y.cols();
    int threads=BLOCK_SIZE;
    int blocks=(size+threads-1)/threads;
    mse_kernel<<<blocks,threads>>>(y_.data(),y.data(),dy.data(),size);
}   

void cross_entropy_cuda(const Tensor& y_, const Tensor& y, Tensor& dy)
{
    int n=y.rows();
    int k=y.cols();
    int size=n*k;
    float epsilon=0.1f;
    int threads=BLOCK_SIZE;
    int blocks=(size+threads-1)/threads;
    cross_entropy_kernel<<<blocks,threads>>>(y_.data(),y.data(),dy.data(),size,n,k,epsilon);
}

float mse_loss_cuda(const Tensor& y_,const Tensor& y,Tensor& loss)
{
    float* d_loss=loss.data();
    cudaMemset(d_loss,0,sizeof(float));

    int size=y.rows()*y.cols();
    int threads=BLOCK_SIZE;
    int blocks=(size+threads-1)/threads;
    mse_loss_kernel<<<blocks,threads>>>(y_.data(),y.data(),d_loss,size);

    cudaDeviceSynchronize();

    float h_loss=0.0f;
    cudaMemcpy(&h_loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost);
    return h_loss;
}

float cross_entropy_loss_cuda(const Tensor& y_, const Tensor& y, Tensor& loss)
{
    float* d_loss = loss.data();
    cudaMemset(d_loss, 0, sizeof(float));

    int size = y.rows() * y.cols();
    int k    = y.cols();
    float epsilon = 0.1f;

    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_entropy_loss_kernel<<<blocks, BLOCK_SIZE>>>(y_.data(), y.data(), d_loss, size, y_.rows(), k, epsilon);

    cudaDeviceSynchronize();

    float h_loss = 0.0f;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss;
}