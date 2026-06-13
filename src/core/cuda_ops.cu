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
    if(k1!=k2) throw std::invalid_argument("Inner dimension must match for matrix multiplication");

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
__global__ void cross_entropy_kernel(const float* pred, const float* targets_idx, float* dy, int batch_seq, int vocab_size, int valid_tokens, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_seq * vocab_size;
    if(idx < total_elements)
    {
        int row = idx / vocab_size;
        int col = idx % vocab_size;
        int target_token = (int)targets_idx[row];
        
        if (target_token == -100) {
            dy[idx] = 0.0f;
        } else {
            float target_val = (col == target_token) ? (1.0f - epsilon) + (epsilon / (float)vocab_size) : (epsilon / (float)vocab_size);
            dy[idx] = (pred[idx] - target_val) / (float)valid_tokens;
        }
    }
}

__global__ void cross_entropy_loss_kernel(const float* pred, const float* targets_idx, float* loss, int batch_seq, int vocab_size, int valid_tokens, float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_seq)
    {
        int target_token = (int)targets_idx[idx];
        if (target_token != -100 && target_token >= 0 && target_token < vocab_size) {
            float target_val = (1.0f - epsilon) + (epsilon / (float)vocab_size);
            atomicAdd(loss, -target_val * logf(pred[idx * vocab_size + target_token] + 1e-7f) / (float)valid_tokens);
            // Technically label smoothing adds terms for all incorrect classes, but for the loss scalar 
            // the main term is sufficient for monitoring. We will add the smooth terms if needed.
            // But since epsilon is small, tracking the true class loss is usually fine.
            // Let's add the smooth terms for correctness:
            float smooth_target = epsilon / (float)vocab_size;
            float smooth_loss = 0.0f;
            for(int v=0; v<vocab_size; ++v) {
                if (v != target_token) {
                    smooth_loss += -smooth_target * logf(pred[idx * vocab_size + v] + 1e-7f);
                }
            }
            atomicAdd(loss, smooth_loss / (float)valid_tokens);
        }
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
        
        // Gradient clipping: clamp per-element gradient to prevent exploding gradients
        float max_grad = 100.0f; // Relaxed to 100.0f to let Adam naturally normalize gradients
        if(g > max_grad) g = max_grad;
        if(g < -max_grad) g = -max_grad;
        
        float mt=0.9f*m[idx]+(1.0f-0.9f)*g;
        float vt=0.999f*v[idx]+(1.0f-0.999f)*g*g;
        
        m[idx]=mt;
        v[idx]=vt;
        
        mt=mt/(1.0f-powf(0.9f,t));
        vt=vt/(1.0f-powf(0.999f, t));
        
        w[idx]-=lr*mt/(sqrtf(vt)+1e-8f);
    }
}

__global__ void clamp_kernel(float* data, float min_val, float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] > max_val) data[idx] = max_val;
        if (data[idx] < min_val) data[idx] = min_val;
    }
}

__global__ void calculate_accuracy_kernel(const float* pred, const float* targets_idx, int* top1, int* top5, int* total, int seq_len, int vocab_size, int total_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_tokens) {
        int target_token = (int)targets_idx[idx];
        if (target_token == -100) return; // Ignore padding and prompts

        atomicAdd(total, 1);

        float top5_vals[5] = {-1e9f, -1e9f, -1e9f, -1e9f, -1e9f};
        int top5_idx[5] = {-1, -1, -1, -1, -1};

        int row_start = idx * vocab_size;
        for (int v = 0; v < vocab_size; ++v) {
            float val = pred[row_start + v];
            for (int k = 0; k < 5; ++k) {
                if (val > top5_vals[k]) {
                    for (int j = 4; j > k; --j) {
                        top5_idx[j] = top5_idx[j-1];
                        top5_vals[j] = top5_vals[j-1];
                    }
                    top5_idx[k] = v;
                    top5_vals[k] = val;
                    break;
                }
            }
        }

        if (top5_idx[0] == target_token) atomicAdd(top1, 1);
        for (int k = 0; k < 5; ++k) {
            if (top5_idx[k] == target_token) {
                atomicAdd(top5, 1);
                break;
            }
        }
    }
}

__global__ void sum_squares_kernel(const float* grad, float* sum_sq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = grad[idx];
        atomicAdd(sum_sq, val * val);
    }
}

__global__ void scale_tensor_kernel(float* grad, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] *= scale;
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

__global__ void sigmoid_forward_kernel(float* Y, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) Y[index] = 1.0f / (1.0f + expf(-Y[index]));
}

__global__ void sigmoid_backward_kernel(const float* dY, const float* cached_X, float* dX, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float s = 1.0f / (1.0f + expf(-cached_X[index]));
        dX[index] = dY[index] * s * (1.0f - s);
    }
}

__global__ void gelu_forward_kernel(float* Y, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float x = Y[index];
        float v = 0.7978845608f * (x + 0.044715f * x * x * x);
        Y[index] = 0.5f * x * (1.0f + tanhf(v));
    }
}

__global__ void gelu_backward_kernel(const float* dY, const float* cached_X, float* dX, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float x = cached_X[index];
        float x3 = x * x * x;
        float v = 0.7978845608f * (x + 0.044715f * x3);
        float t = tanhf(v);
        float sech2 = 1.0f - t * t;
        float dv_dx = 0.7978845608f * (1.0f + 0.134145f * x * x);
        dX[index] = dY[index] * (0.5f * (1.0f + t) + 0.5f * x * sech2 * dv_dx);
    }
}

__global__ void tanh_forward_kernel(float* Y, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) Y[index] = tanhf(Y[index]);
}

__global__ void tanh_backward_kernel(const float* dY, const float* cached_X, float* dX, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float t = tanhf(cached_X[index]);
        dX[index] = dY[index] * (1.0f - t * t);
    }
}

__global__ void relu_forward_kernel(float* Y, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) Y[index] = Y[index] > 0.0f ? Y[index] : 0.0f;
}

__global__ void relu_backward_kernel(const float* dY, const float* cached_X, float* dX, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) dX[index] = cached_X[index] > 0.0f ? dY[index] : 0.0f;
}

__global__ void silu_forward_kernel(float* Y, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float x = Y[index];
        Y[index] = x / (1.0f + expf(-x));
    }
}

__global__ void silu_backward_kernel(const float* dY, const float* cached_X, float* dX, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        float x = cached_X[index];
        float s = 1.0f / (1.0f + expf(-x));
        dX[index] = dY[index] * (s * (1.0f + x * (1.0f - s)));
    }
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
        unsigned int s = (unsigned int)seed ^ ((unsigned int)index * 1664525 + 1013904223);
        s ^= s >> 16; s *= 0x85ebca6b;
        s ^= s >> 13; s *= 0xc2b2ae35;
        s ^= s >> 16;
        float val = (float)s / 4294967296.0f;
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

__global__ void embedding_forward_kernel(const float* input,const float* w,float* output,int tokens,int dimension,int size)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<tokens*dimension)
    {
        int token_index=index/dimension;
        int dim_index=index%dimension;

        int id=(int)input[token_index];
        if(id>=0 && id<size)output[index]=w[id*dimension+dim_index];
        else output[index]=0.0f;
    }
}

__global__ void embedding_backward_kernel(const float* input,const float* dY,float *dW,int tokens,int dimension,int size)
{
    int index=blockDim.x*blockIdx.x+threadIdx.x;
    if(index<tokens*dimension)
    {
        int token_index=index/dimension;
        int dim_index=index%dimension;

        int id=(int)input[token_index];
        if(id>=0 && id<size) atomicAdd(&dW[id*dimension+dim_index],dY[index]);
    }
}

__global__ void attention_scale_kernel(float* data,float scale,int size)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size) data[idx]*=scale;
}

__global__ void attention_softmax_kernel(float* data,int rows,int cols)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows)
    {
        float* row_ptr=data+row*cols;
        int token_pos=row%cols;

        // Causal mask: set future positions to -inf
        for(int i=token_pos+1;i<cols;i++) row_ptr[i]=-1e38f;

        float max_val=row_ptr[0];
        for(int i=1;i<=token_pos;i++) if(row_ptr[i]>max_val) max_val=row_ptr[i];

        float sum=0.0f;
        for(int i=0;i<=token_pos;i++)
        {
            row_ptr[i]=expf(row_ptr[i]-max_val);
            sum+=row_ptr[i];
        }

        float inv_sum=1.0f/sum;
        for(int i=0;i<=token_pos;i++) row_ptr[i]*=inv_sum;
        for(int i=token_pos+1;i<cols;i++) row_ptr[i]=0.0f;
    }
}

__global__ void attention_softmax_backward_kernel(const float* dY,const float* Y,float* dX,int rows,int cols)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows)
    {
        int offset=row*cols;
        int token_pos=row%cols;

        float dot=0.0f;
        for(int i=0;i<=token_pos;i++) dot+=dY[offset+i]*Y[offset+i];

        for(int i=0;i<=token_pos;i++) dX[offset+i]=Y[offset+i]*(dY[offset+i]-dot);
        for(int i=token_pos+1;i<cols;i++) dX[offset+i]=0.0f;
    }
}

__global__ void full_attention_softmax_kernel(float* data,int rows,int cols)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows)
    {
        float* row_ptr=data+row*cols;

        float max_val=row_ptr[0];
        for(int i=1;i<cols;i++) if(row_ptr[i]>max_val) max_val=row_ptr[i];

        float sum=0.0f;
        for(int i=0;i<cols;i++)
        {
            row_ptr[i]=expf(row_ptr[i]-max_val);
            sum+=row_ptr[i];
        }

        float inv_sum=1.0f/sum;
        for(int i=0;i<cols;i++) row_ptr[i]*=inv_sum;
    }
}

__global__ void full_attention_softmax_backward_kernel(const float* dY,const float* Y,float* dX,int rows,int cols)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows)
    {
        int offset=row*cols;

        float dot=0.0f;
        for(int i=0;i<cols;i++) dot+=dY[offset+i]*Y[offset+i];

        for(int i=0;i<cols;i++) dX[offset+i]=Y[offset+i]*(dY[offset+i]-dot);
    }
}

__global__ void split_heads_kernel(const float* input,float* output,int N,int T,int H,int dk)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=N*H*T*dk;
    if(idx<total)
    {
        int d=idx%dk;
        int t=(idx/dk)%T;
        int h=(idx/(dk*T))%H;
        int n=idx/(dk*T*H);

        int in_idx=(n*T+t)*(H*dk)+h*dk+d;
        output[idx]=input[in_idx];
    }
}

__global__ void merge_heads_kernel(const float* input,float* output,int N,int T,int H,int dk)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int total=N*T*H*dk;
    if(idx<total)
    {
        int D=H*dk;
        int d=idx%D;
        int nt=idx/D;

        int h=d/dk;
        int di=d%dk;
        int n=nt/T;
        int t=nt%T;

        int in_idx=(n*H+h)*T*dk+t*dk+di;
        output[idx]=input[in_idx];
    }
}

__global__ void layernorm_forward_kernel(const float* X,const float* g,const float* b,float* Y,float* mean,float* var,float* x_hat,int rows,int cols,float eps)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows)
    {
        int offset=row*cols;

        float sum=0.0f;
        for(int i=0;i<cols;i++) sum+=X[offset+i];
        float m=sum/(float)cols;
        mean[row]=m;

        float sq_sum=0.0f;
        for(int i=0;i<cols;i++) sq_sum+=(X[offset+i]-m)*(X[offset+i]-m);
        float v=sq_sum/(float)cols;
        var[row]=v;

        float inv_std=rsqrtf(v+eps);
        for(int i=0;i<cols;i++)
        {
            float xh=(X[offset+i]-m)*inv_std;
            x_hat[offset+i]=xh;
            Y[offset+i]=g[i]*xh+b[i];
        }
    }
}

__global__ void layernorm_backward_kernel(const float* dY,const float* x_hat,const float* g,const float* var,float* dX,float* dg,float* db,int rows,int cols,float eps)
{
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<rows)
    {
        int offset=row*cols;
        float inv_std=rsqrtf(var[row]+eps);

        float dot_dy_xhat=0.0f;
        float dot_dy=0.0f;
        for(int i=0;i<cols;i++)
        {
            float dy_scaled=dY[offset+i]*g[i];
            dot_dy_xhat+=dy_scaled*x_hat[offset+i];
            dot_dy+=dy_scaled;
        }

        float inv_cols=1.0f/(float)cols;
        for(int i=0;i<cols;i++)
        {
            float dy_scaled=dY[offset+i]*g[i];
            dX[offset+i]=inv_std*(dy_scaled-inv_cols*(dot_dy+x_hat[offset+i]*dot_dy_xhat));

            atomicAdd(&dg[i],dY[offset+i]*x_hat[offset+i]);
            atomicAdd(&db[i],dY[offset+i]);
        }
    }
}

void attention_scale_cuda(float* data,float scale,int size)
{
    int blocks=(size+BLOCK_SIZE-1)/BLOCK_SIZE;
    attention_scale_kernel<<<blocks,BLOCK_SIZE>>>(data,scale,size);
}

void attention_softmax_cuda(float* data,int rows,int cols)
{
    int blocks=(rows+BLOCK_SIZE-1)/BLOCK_SIZE;
    attention_softmax_kernel<<<blocks,BLOCK_SIZE>>>(data,rows,cols);
}

void attention_softmax_backward_cuda(const float* dY,const float* Y,float* dX,int rows,int cols)
{
    int blocks=(rows+BLOCK_SIZE-1)/BLOCK_SIZE;
    attention_softmax_backward_kernel<<<blocks,BLOCK_SIZE>>>(dY,Y,dX,rows,cols);
}

void full_attention_softmax_cuda(float* data,int rows,int cols)
{
    int blocks=(rows+BLOCK_SIZE-1)/BLOCK_SIZE;
    full_attention_softmax_kernel<<<blocks,BLOCK_SIZE>>>(data,rows,cols);
}

void full_attention_softmax_backward_cuda(const float* dY,const float* Y,float* dX,int rows,int cols)
{
    int blocks=(rows+BLOCK_SIZE-1)/BLOCK_SIZE;
    full_attention_softmax_backward_kernel<<<blocks,BLOCK_SIZE>>>(dY,Y,dX,rows,cols);
}

void split_heads_cuda(const float* input,float* output,int N,int T,int H,int dk)
{
    int total=N*H*T*dk;
    int blocks=(total+BLOCK_SIZE-1)/BLOCK_SIZE;
    split_heads_kernel<<<blocks,BLOCK_SIZE>>>(input,output,N,T,H,dk);
}

void merge_heads_cuda(const float* input,float* output,int N,int T,int H,int dk)
{
    int total=N*T*H*dk;
    int blocks=(total+BLOCK_SIZE-1)/BLOCK_SIZE;
    merge_heads_kernel<<<blocks,BLOCK_SIZE>>>(input,output,N,T,H,dk);
}

void batched_matmul_cuda(const float* A,bool transA,const float* B,bool transB,float* C,int batch,int M,int K,int N)
{
    cublasHandle_t handle=CudaContext::get_instance().get_cublas_handle();

    cublasOperation_t opA_cublas=transA?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t opB_cublas=transB?CUBLAS_OP_T:CUBLAS_OP_N;

    int lda_cublas=transA?M:K;    
    int ldb_cublas=transB?K:N;   
    int ldc_cublas=N;             

    int strideA_rows=transA?K:M;
    int strideA_cols=transA?M:K;
    long long strideA=(long long)strideA_rows*strideA_cols;
    int strideB_rows=transB?N:K;
    int strideB_cols=transB?K:N;
    long long strideB=(long long)strideB_rows*strideB_cols;
    long long strideC=(long long)M*N;

    float alpha=1.0f;
    float beta=0.0f;

    cublasSgemmStridedBatched(
        handle,
        opB_cublas,     
        opA_cublas,     
        N,M,K,          
        &alpha,
        B,ldb_cublas,strideB,
        A,lda_cublas,strideA,
        &beta,
        C,ldc_cublas,strideC,
        batch
    );
}

void embedding_forward_cuda(const float* X,const float* dW,float* Y,int tokens,int dimension,int size)
{
    int threads=BLOCK_SIZE;
    int blocks=(tokens*dimension+threads-1)/threads;
    embedding_forward_kernel<<<blocks,threads>>>(X,dW,Y,tokens,dimension,size);   
}

void embedding_backward_cuda(const float* X,const float* dY,float* dW,int tokens,int dimension,int size)
{
    int threads=BLOCK_SIZE;
    int blocks=(tokens*dimension+threads-1)/threads;
    embedding_backward_kernel<<<blocks,threads>>>(X,dY,dW,tokens,dimension,size);
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

void sigmoid_forward_cuda(Tensor& Y)
{
    int size = Y.rows() * Y.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    sigmoid_forward_kernel<<<blocks, threads>>>(Y.data(), size);
}

void sigmoid_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX)
{
    int size = dY.rows() * dY.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(dY.data(), cached_X.data(), dX.data(), size);
}

void gelu_forward_cuda(Tensor& Y)
{
    int size = Y.rows() * Y.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    gelu_forward_kernel<<<blocks, threads>>>(Y.data(), size);
}

void gelu_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX)
{
    int size = dY.total_elements();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_backward_kernel<<<blocks, threads>>>(dY.data(), cached_X.data(), dX.data(), size);
}

void clamp_tensor_cuda(float* data, float min_val, float max_val, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    clamp_kernel<<<blocks, threads>>>(data, min_val, max_val, size);
}

// Reads the precomputed sum-of-squares (device scalar) and scales grad in
// place by min(1, max_norm/norm); non-finite norm -> 0. Each thread recomputes
// the (cheap) sqrt so no host round-trip or broadcast is needed.
__global__ void apply_grad_clip_kernel(float* grad, const float* d_sum_sq, float max_norm, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float norm = sqrtf(d_sum_sq[0]);
        float scale = 1.0f;
        if (!isfinite(norm)) scale = 0.0f;
        else if (norm > max_norm) scale = max_norm / norm;
        grad[idx] *= scale;
    }
}

void clip_grad_norm_tensor_cuda(Tensor& grad, float max_norm) {
    int size = grad.total_elements();
    if (size == 0) return;

    // Persistent device scalar reused across all calls. The previous version
    // did cudaMalloc + cudaDeviceSynchronize + D2H cudaMemcpy + cudaFree on
    // EVERY call -- and grad-clip fires ~170x per training step. cudaMalloc/
    // cudaFree are device-synchronizing, so that was ~170 full pipeline drains
    // per batch (the dominant training cost). Everything stays on the default
    // stream, so memset -> sum_squares -> apply_clip remain correctly ordered
    // without any host synchronization.
    static float* d_sum_sq = nullptr;
    if (!d_sum_sq) cudaMalloc(&d_sum_sq, sizeof(float));
    cudaMemsetAsync(d_sum_sq, 0, sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sum_squares_kernel<<<blocks, threads>>>(grad.data(), d_sum_sq, size);
    apply_grad_clip_kernel<<<blocks, threads>>>(grad.data(), d_sum_sq, max_norm, size);
}

void calculate_accuracy_cuda(const float* pred, const float* targets_idx, int* top1_out, int* top5_out, int* total_out, int seq_len, int vocab_size, int total_tokens) {
    int* d_top1;
    int* d_top5;
    int* d_total;
    cudaMalloc(&d_top1, sizeof(int));
    cudaMalloc(&d_top5, sizeof(int));
    cudaMalloc(&d_total, sizeof(int));
    cudaMemset(d_top1, 0, sizeof(int));
    cudaMemset(d_top5, 0, sizeof(int));
    cudaMemset(d_total, 0, sizeof(int));

    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;
    calculate_accuracy_kernel<<<blocks, threads>>>(pred, targets_idx, d_top1, d_top5, d_total, seq_len, vocab_size, total_tokens);
    cudaDeviceSynchronize();

    cudaMemcpy(top1_out, d_top1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(top5_out, d_top5, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(total_out, d_total, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_top1);
    cudaFree(d_top5);
    cudaFree(d_total);
}

void tanh_forward_cuda(Tensor& Y)
{
    int size = Y.rows() * Y.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    tanh_forward_kernel<<<blocks, threads>>>(Y.data(), size);
}

void tanh_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX)
{
    int size = dY.rows() * dY.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    tanh_backward_kernel<<<blocks, threads>>>(dY.data(), cached_X.data(), dX.data(), size);
}

void relu_forward_cuda(Tensor& Y)
{
    int size = Y.rows() * Y.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(Y.data(), size);
}

void relu_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX)
{
    int size = dY.rows() * dY.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(dY.data(), cached_X.data(), dX.data(), size);
}

void silu_forward_cuda(Tensor& Y)
{
    int size = Y.rows() * Y.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    silu_forward_kernel<<<blocks, threads>>>(Y.data(), size);
}

void silu_backward_cuda(const Tensor& dY, const Tensor& cached_X, Tensor& dX)
{
    int size = dY.rows() * dY.cols();
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    silu_backward_kernel<<<blocks, threads>>>(dY.data(), cached_X.data(), dX.data(), size);
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

// ---- Masked grouped mean pooling (collapse schema sub-tokens into one vector per element) ----
// in: [num_groups*group_size, dim]; tokens: [num_groups*group_size] token ids; out: [num_groups, dim].
// Averages over the NON-pad sub-tokens of each group only, so short schema elements aren't diluted
// by the (learned) pad-row vector. A group with zero real tokens yields 0.
__global__ void mean_pool_groups_kernel(const float* in, const float* tokens, float* out, int num_groups, int group_size, int dim, float pad_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_groups * dim;
    if (idx < total) {
        int g = idx / dim;
        int d = idx % dim;
        float s = 0.0f; int cnt = 0;
        for (int k = 0; k < group_size; ++k) {
            if (tokens[g * group_size + k] != pad_id) { s += in[(g * group_size + k) * dim + d]; cnt++; }
        }
        out[idx] = (cnt > 0) ? s / (float)cnt : 0.0f;
    }
}

// Broadcast each group's gradient back across its NON-pad sub-tokens (divided by the real count);
// pad sub-tokens get zero gradient (they didn't contribute to the pooled vector).
__global__ void mean_pool_groups_backward_kernel(const float* dout, const float* tokens, float* din, int num_groups, int group_size, int dim, float pad_id)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_groups * group_size * dim;
    if (idx < total) {
        int row = idx / dim;     // = g*group_size + k
        int d = idx % dim;
        int k = row % group_size;
        int g = row / group_size;
        if (tokens[g * group_size + k] == pad_id) { din[idx] = 0.0f; return; }
        int cnt = 0;
        for (int kk = 0; kk < group_size; ++kk) if (tokens[g * group_size + kk] != pad_id) cnt++;
        din[idx] = (cnt > 0) ? dout[g * dim + d] / (float)cnt : 0.0f;
    }
}

void mean_pool_groups_cuda(const float* in, const float* tokens, float* out, int num_groups, int group_size, int dim, float pad_id)
{
    int total = num_groups * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_pool_groups_kernel<<<blocks, threads>>>(in, tokens, out, num_groups, group_size, dim, pad_id);
}

void mean_pool_groups_backward_cuda(const float* dout, const float* tokens, float* din, int num_groups, int group_size, int dim, float pad_id)
{
    int total = num_groups * group_size * dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_pool_groups_backward_kernel<<<blocks, threads>>>(dout, tokens, din, num_groups, group_size, dim, pad_id);
}

// Average attention over heads: attn [N*heads, Tq, S] -> attn_mean [N*Tq, S].
__global__ void reduce_heads_attn_kernel(const float* attn, float* out, int batch, int heads, int Tq, int S)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * Tq * S;
    if (idx < total) {
        int j = idx % S;
        int t = (idx / S) % Tq;
        int n = idx / (S * Tq);
        float s = 0.0f;
        for (int h = 0; h < heads; ++h)
            s += attn[((n * heads + h) * Tq + t) * S + j];
        out[(n * Tq + t) * S + j] = s / (float)heads;
    }
}

// Add -1e9 to padded schema elements in the attention scores
__global__ void apply_attention_mask_kernel(float* scores, const float* mask, int batch, int heads, int Tq, int S)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * Tq * S;
    if (idx < total) {
        int j = idx % S;
        int n = idx / (S * Tq * heads);
        // mask is shape [batch, S]
        if (mask[n * S + j] < 0.5f) {
            scores[idx] += -1e9f;
        }
    }
}

// Inverse: spread d_attn_mean [N*Tq, S] back to all heads (divided by heads).
__global__ void expand_heads_attn_kernel(const float* d_mean, float* d_attn, int batch, int heads, int Tq, int S)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * Tq * S;
    if (idx < total) {
        int j = idx % S;
        int t = (idx / S) % Tq;
        int h = (idx / (S * Tq)) % heads;
        int n = idx / (S * Tq * heads);
        d_attn[((n * heads + h) * Tq + t) * S + j] = d_mean[(n * Tq + t) * S + j] / (float)heads;
    }
}

// Scatter schema attention into the vocab: P_schema[bt, vocab_id[j]] += attn_mean[bt, j].
__global__ void copy_scatter_kernel(const float* attn_mean, const float* vocab_ids, float* P_schema, int batch_seq, int S, int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * S;
    if (idx < total) {
        int bt = idx / S;
        int j = idx % S;
        int v = (int)vocab_ids[j];
        if (v >= 0 && v < V)
            atomicAdd(&P_schema[bt * V + v], attn_mean[bt * S + j]);
    }
}

// Gather back: d_attn_mean[bt, j] = dP_schema[bt, vocab_id[j]].
__global__ void copy_gather_kernel(const float* dP_schema, const float* vocab_ids, float* d_attn_mean, int batch_seq, int S, int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * S;
    if (idx < total) {
        int bt = idx / S;
        int j = idx % S;
        int v = (int)vocab_ids[j];
        d_attn_mean[idx] = (v >= 0 && v < V) ? dP_schema[bt * V + v] : 0.0f;
    }
}

// P_final = p_gen * P_vocab + (1 - p_gen) * P_schema   (p_gen broadcast over vocab).
__global__ void blend_forward_kernel(const float* p_gen, const float* P_vocab, const float* P_schema, float* P_final, int batch_seq, int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * V;
    if (idx < total) {
        int bt = idx / V;
        float pg = p_gen[bt];
        P_final[idx] = pg * P_vocab[idx] + (1.0f - pg) * P_schema[idx];
    }
}

// True cross-entropy-on-probability gradient (label-smoothed, masked): dL/dP_final.
__global__ void ce_prob_grad_kernel(const float* P_final, const float* targets_idx, float* dP, int batch_seq, int V, int valid_tokens, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * V;
    if (idx < total) {
        int row = idx / V;
        int col = idx % V;
        int target = (int)targets_idx[row];
        if (target == -100) { dP[idx] = 0.0f; return; }
        float t_v = (col == target) ? (1.0f - eps) + (eps / (float)V) : (eps / (float)V);
        // The true CE-on-probability gradient -t/P. Now that the schema mask ensures
        // P_final is a valid probability distribution, we can relax the per-element clamp 
        // to preserve the exact (pred - target) cancellation through the Softmax Jacobians.
        // We floor P at 1e-7 to prevent literal Inf/NaNs, and rely on global gradient norm clipping.
        float p = fmaxf(P_final[idx], 1e-7f);
        float g = -t_v / p / (float)valid_tokens;
        dP[idx] = g;
    }
}

// Split the blend gradient into the two branches.
__global__ void blend_backward_dist_kernel(const float* p_gen, const float* dP_final, float* dP_vocab, float* dP_schema, int batch_seq, int V)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_seq * V;
    if (idx < total) {
        int bt = idx / V;
        float pg = p_gen[bt];
        float g = dP_final[idx];
        dP_vocab[idx] = pg * g;
        dP_schema[idx] = (1.0f - pg) * g;
    }
}

__global__ void blend_backward_pgen_kernel(const float* p_gen, const float* targets_idx, float* dp_gen, int batch_seq, int valid_tokens)
{
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    if (bt < batch_seq) {
        int target = (int)targets_idx[bt];
        if (target == -100) {
            dp_gen[bt] = 0.0f;
            return;
        }
        
        float pg = p_gen[bt];
        pg = fmaxf(1e-7f, fminf(1.0f - 1e-7f, pg));
        
        float g = 0.0f;
        if (target < 50000) {
            // Target is vocab, optimal p_gen is 1.0. Derivative of -log(p_gen)
            g = -1.0f / pg;
        } else {
            // Target is schema, optimal p_gen is 0.0. Derivative of -log(1 - p_gen)
            g = 1.0f / (1.0f - pg);
        }
        
        dp_gen[bt] = g / (float)valid_tokens;
    }
}

// (reuses the existing sigmoid_forward_kernel defined earlier in this file)

// dy *= s*(1-s)  (s = sigmoid output)
__global__ void sigmoid_grad_mul_kernel(const float* s, float* dy, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) dy[idx] *= s[idx] * (1.0f - s[idx]);
}

void reduce_heads_attn_cuda(const float* attn, float* out, int batch, int heads, int Tq, int S)
{
    int total = batch * Tq * S; int th = 256; int bl = (total + th - 1) / th;
    reduce_heads_attn_kernel<<<bl, th>>>(attn, out, batch, heads, Tq, S);
}
void apply_attention_mask_cuda(float* scores, const float* mask, int batch, int heads, int Tq, int S)
{
    int total = batch * heads * Tq * S; int th = 256; int bl = (total + th - 1) / th;
    apply_attention_mask_kernel<<<bl, th>>>(scores, mask, batch, heads, Tq, S);
}
void expand_heads_attn_cuda(const float* d_mean, float* d_attn, int batch, int heads, int Tq, int S)
{
    int total = batch * heads * Tq * S; int th = 256; int bl = (total + th - 1) / th;
    expand_heads_attn_kernel<<<bl, th>>>(d_mean, d_attn, batch, heads, Tq, S);
}
void copy_scatter_cuda(const float* attn_mean, const float* vocab_ids, float* P_schema, int batch_seq, int S, int V)
{
    cudaMemset(P_schema, 0, (size_t)batch_seq * V * sizeof(float));
    int total = batch_seq * S; int th = 256; int bl = (total + th - 1) / th;
    copy_scatter_kernel<<<bl, th>>>(attn_mean, vocab_ids, P_schema, batch_seq, S, V);
}
void copy_gather_cuda(const float* dP_schema, const float* vocab_ids, float* d_attn_mean, int batch_seq, int S, int V)
{
    int total = batch_seq * S; int th = 256; int bl = (total + th - 1) / th;
    copy_gather_kernel<<<bl, th>>>(dP_schema, vocab_ids, d_attn_mean, batch_seq, S, V);
}
void blend_forward_cuda(const float* p_gen, const float* P_vocab, const float* P_schema, float* P_final, int batch_seq, int V)
{
    int total = batch_seq * V; int th = 256; int bl = (total + th - 1) / th;
    blend_forward_kernel<<<bl, th>>>(p_gen, P_vocab, P_schema, P_final, batch_seq, V);
}
void ce_prob_grad_cuda(const float* P_final, const float* targets_idx, float* dP, int batch_seq, int V, int valid_tokens)
{
    int total = batch_seq * V; int th = 256; int bl = (total + th - 1) / th;
    ce_prob_grad_kernel<<<bl, th>>>(P_final, targets_idx, dP, batch_seq, V, valid_tokens, 0.1f);
}
void blend_backward_dist_cuda(const float* p_gen, const float* dP_final, float* dP_vocab, float* dP_schema, int batch_seq, int V)
{
    int total = batch_seq * V; int th = 256; int bl = (total + th - 1) / th;
    blend_backward_dist_kernel<<<bl, th>>>(p_gen, dP_final, dP_vocab, dP_schema, batch_seq, V);
}
void blend_backward_pgen_cuda(const float* p_gen, const float* targets_idx, float* dp_gen, int batch_seq, int valid_tokens)
{
    int th = 256; int bl = (batch_seq + th - 1) / th;
    blend_backward_pgen_kernel<<<bl, th>>>(p_gen, targets_idx, dp_gen, batch_seq, valid_tokens);
}
void sigmoid_forward_cuda(float* data, int size)
{
    int th = 256; int bl = (size + th - 1) / th;
    sigmoid_forward_kernel<<<bl, th>>>(data, size);
}
void sigmoid_grad_mul_cuda(const float* s, float* dy, int size)
{
    int th = 256; int bl = (size + th - 1) / th;
    sigmoid_grad_mul_kernel<<<bl, th>>>(s, dy, size);
}

__global__ void gate_entropy_regularization_kernel(float* dp_gen, const float* p_gen, float lambda, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dp_gen[idx] += lambda * (2.0f * p_gen[idx] - 1.0f);
    }
}

void gate_entropy_regularization_cuda(float* dp_gen, const float* p_gen, float lambda, int size)
{
    int th = 256; int bl = (size + th - 1) / th;
    gate_entropy_regularization_kernel<<<bl, th>>>(dp_gen, p_gen, lambda, size);
}

__global__ void vocab_logits_grad_kernel(const float* P_vocab, const float* targets_idx, float* d_logits, int batch_seq, int V, int valid_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_seq * V) {
        int bt = idx / V;
        int v = idx % V;
        int target = (int)targets_idx[bt];
        if (target == -100 || target >= 50000) {
            d_logits[idx] = 0.0f;
        } else {
            float t_v = (v == target) ? 1.0f : 0.0f;
            d_logits[idx] = (P_vocab[idx] - t_v) / (float)valid_tokens;
        }
    }
}

void vocab_logits_grad_cuda(const float* P_vocab, const float* targets_idx, float* d_logits, int batch_seq, int V, int valid_tokens) {
    int total = batch_seq * V;
    int th = 256; int bl = (total + th - 1) / th;
    vocab_logits_grad_kernel<<<bl, th>>>(P_vocab, targets_idx, d_logits, batch_seq, V, valid_tokens);
}

__global__ void attn_logits_grad_kernel(const float* P_attn, const float* P_attn_mean, const float* schema_vocab_ids, const float* targets_idx, float* d_logits, int batch, int heads, int seq, int S, int valid_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * heads * seq * S;
    if (idx < total) {
        int s = idx % S;
        int t = (idx / S) % seq;
        int h = (idx / (S * seq)) % heads;
        int n = idx / (S * seq * heads);
        int bt = n * seq + t;
        
        int target = (int)targets_idx[bt];
        if (target == -100 || target < 50000) {
            d_logits[idx] = 0.0f;
            return;
        }
        
        int target_s = -1;
        for (int i = 0; i < S; ++i) {
            if ((int)schema_vocab_ids[i] == target) {
                target_s = i;
                break;
            }
        }
        
        if (target_s == -1) {
            d_logits[idx] = 0.0f;
            return;
        }
        
        float t_s = (s == target_s) ? 1.0f : 0.0f;
        
        // We bypass the exact Softmax Jacobian ratio (R = P_h / P_mean).
        // If a head has 0 probability for the target, R=0 and it receives NO gradient to correct itself (vanishing gradient).
        // By replacing it with a flat 1/heads, we treat this as the Mean of Cross Entropies instead of Cross Entropy of the Mean.
        // This provides a massive, pristine unattenuated gradient to EVERY head that is pointing at the wrong token!
        float g = (P_attn[idx] - t_s) / (float)heads;
        
        d_logits[idx] = g / (float)valid_tokens;
    }
}

void attn_logits_grad_cuda(const float* P_attn, const float* P_attn_mean, const float* schema_vocab_ids, const float* targets_idx, float* d_logits, int batch, int heads, int seq, int S, int valid_tokens) {
    int total = batch * heads * seq * S;
    int th = 256; int bl = (total + th - 1) / th;
    attn_logits_grad_kernel<<<bl, th>>>(P_attn, P_attn_mean, schema_vocab_ids, targets_idx, d_logits, batch, heads, seq, S, valid_tokens);
}

void cross_entropy_cuda(const float* pred, const float* targets_idx, float* dy, int batch_seq, int vocab_size, int valid_tokens)
{
    int total_elements = batch_seq * vocab_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    cross_entropy_kernel<<<blocks, threads>>>(pred, targets_idx, dy, batch_seq, vocab_size, valid_tokens, 0.1f);
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

float cross_entropy_loss_cuda(const float* pred, const float* targets_idx, float* loss, int batch_seq, int vocab_size, int valid_tokens)
{
    cudaMemset(loss, 0, sizeof(float));

    // One thread per token; the kernel iterates the vocab internally and skips -100 targets.
    int threads = 256;
    int blocks = (batch_seq + threads - 1) / threads;
    cross_entropy_loss_kernel<<<blocks, threads>>>(pred, targets_idx, loss, batch_seq, vocab_size, valid_tokens, 0.1f);
    cudaDeviceSynchronize();

    float h_loss = 0.0f;
    cudaMemcpy(&h_loss, loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss;
}

// ---- Dense one-hot cross entropy (legacy path for Network / ViT / BERT / timeseries) ----
// Targets are full one-hot rows matching predictions; divide by row count (no ignore mask).
__global__ void cross_entropy_dense_kernel(const float* y_, const float* y, float* dy, int size, int n, int k, float epsilon)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size)
    {
        float target = y[index] * (1.0f - epsilon) + (epsilon / (float)k);
        dy[index] = (y_[index] - target) / (float)n;
    }
}

__global__ void cross_entropy_loss_dense_kernel(const float* y_, const float* y, float* loss, int size, int n, int k, float epsilon)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < size)
    {
        float target = y[index] * (1.0f - epsilon) + (epsilon / (float)k);
        atomicAdd(loss, -target * logf(y_[index] + 1e-7f) / (float)n);
    }
}

void cross_entropy_dense_cuda(const Tensor& y_, const Tensor& y, Tensor& dy)
{
    int n = y.rows();
    int k = y.cols();
    int size = n * k;
    float epsilon = 0.1f;
    int threads = BLOCK_SIZE;
    int blocks = (size + threads - 1) / threads;
    cross_entropy_dense_kernel<<<blocks, threads>>>(y_.data(), y.data(), dy.data(), size, n, k, epsilon);
}

float cross_entropy_loss_dense_cuda(const Tensor& y_, const Tensor& y, Tensor& loss)
{
    float* d_loss = loss.data();
    cudaMemset(d_loss, 0, sizeof(float));

    int n = y.rows();
    int k = y.cols();
    int size = n * k;
    float epsilon = 0.1f;
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cross_entropy_loss_dense_kernel<<<blocks, BLOCK_SIZE>>>(y_.data(), y.data(), d_loss, size, n, k, epsilon);
    cudaDeviceSynchronize();

    float h_loss = 0.0f;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h_loss;
}

void layernorm_forward_cuda(const float* X,const float* g,const float* b,float* Y,float* mean,float* var,float* x_hat,int rows,int cols,float eps)
{
    int blocks=(rows+BLOCK_SIZE-1)/BLOCK_SIZE;
    layernorm_forward_kernel<<<blocks,BLOCK_SIZE>>>(X,g,b,Y,mean,var,x_hat,rows,cols,eps);
}

void layernorm_backward_cuda(const float* dY,const float* x_hat,const float* g,const float* var,float* dX,float* dg,float* db,int rows,int cols,float eps)
{
    int blocks=(rows+BLOCK_SIZE-1)/BLOCK_SIZE;
    layernorm_backward_kernel<<<blocks,BLOCK_SIZE>>>(dY,x_hat,g,var,dX,dg,db,rows,cols,eps);
}

/*
Mean Pooling: average all tokens in the sequence for each batch.
seq: [batch * seq_len * dim] -> pool: [batch * dim]
*/
__global__ void mean_pool_kernel(const float* seq,float* pool,int batch,int seq_len,int dim)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<batch*dim)
    {
        int b=idx/dim;
        int d=idx%dim;
        float sum=0.0f;
        for(int t=0;t<seq_len;++t)
        {
            sum+=seq[b*seq_len*dim+t*dim+d];
        }
        pool[idx]=sum/seq_len;
    }
}

/*
Mean Pooling Backward: distribute gradient equally to all tokens.
pool: [batch * dim] -> seq: [batch * seq_len * dim]
*/
__global__ void mean_pool_backward_kernel(const float* pool,float* seq,int batch,int seq_len,int dim)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<batch*seq_len*dim)
    {
        int b=idx/(seq_len*dim);
        int d=idx%dim;
        seq[idx]=pool[b*dim+d]/seq_len;
    }
}

void mean_pool_cuda(const float* seq,float* pool,int batch,int seq_len,int dim)
{
    int total=batch*dim;
    int blocks=(total+BLOCK_SIZE-1)/BLOCK_SIZE;
    mean_pool_kernel<<<blocks,BLOCK_SIZE>>>(seq,pool,batch,seq_len,dim);
}

void mean_pool_backward_cuda(const float* pool,float* seq,int batch,int seq_len,int dim)
{
    int total=batch*seq_len*dim;
    int blocks=(total+BLOCK_SIZE-1)/BLOCK_SIZE;
    mean_pool_backward_kernel<<<blocks,BLOCK_SIZE>>>(pool,seq,batch,seq_len,dim);
}

// -------------------------------------------------------------------------
// Pointer-Generator Kernels
// -------------------------------------------------------------------------

__global__ void average_heads_kernel(const float* attn, float* averaged, int N, int H, int T_q, int T_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * T_q * T_k;
    if (idx < total) {
        int n = idx / (T_q * T_k);
        int rem = idx % (T_q * T_k);
        int tq = rem / T_k;
        int tk = rem % T_k;
        
        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            int in_idx = ((n * H + h) * T_q + tq) * T_k + tk;
            sum += attn[in_idx];
        }
        averaged[idx] = sum / H;
    }
}

extern "C" void average_heads_cuda(const float* attn, float* averaged, int N, int H, int T_q, int T_k) {
    int total = N * T_q * T_k;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    average_heads_kernel<<<numBlocks, BLOCK_SIZE>>>(attn, averaged, N, H, T_q, T_k);
}

__global__ void broadcast_heads_kernel(const float* d_averaged, float* d_attn, int N, int H, int T_q, int T_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * T_q * T_k;
    if (idx < total) {
        int n = idx / (H * T_q * T_k);
        int rem = idx % (H * T_q * T_k);
        // int h = rem / (T_q * T_k); // not used directly in out_idx
        int rem2 = rem % (T_q * T_k);
        int tq = rem2 / T_k;
        int tk = rem2 % T_k;
        
        int out_idx = (n * T_q + tq) * T_k + tk;
        d_attn[idx] = d_averaged[out_idx] / H;
    }
}

extern "C" void broadcast_heads_cuda(const float* d_averaged, float* d_attn, int N, int H, int T_q, int T_k) {
    int total = N * H * T_q * T_k;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    broadcast_heads_kernel<<<numBlocks, BLOCK_SIZE>>>(d_averaged, d_attn, N, H, T_q, T_k);
}

__global__ void pointer_blend_forward_kernel(const float* p_vocab, const float* p_copy, const float* p_gen, const float* schema_ids, float* p_final, int N, int T_q, int vocab_size, int T_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * T_q * vocab_size;
    if (idx < total) {
        int n = idx / (T_q * vocab_size);
        int rem = idx % (T_q * vocab_size);
        int tq = rem / vocab_size;
        int v_id = rem % vocab_size;
        
        float gen_prob = p_gen[n * T_q + tq];
        float copy_prob = 0.0f;
        
        // Sum probabilities from all schema elements that map to this vocab ID
        for (int k = 0; k < T_k; ++k) {
            int schema_vocab_id = (int)schema_ids[n * T_k + k];
            if (schema_vocab_id == v_id) {
                copy_prob += p_copy[(n * T_q + tq) * T_k + k];
            }
        }
        
        p_final[idx] = gen_prob * p_vocab[idx] + (1.0f - gen_prob) * copy_prob;
    }
}

extern "C" void pointer_blend_forward_cuda(const float* p_vocab, const float* p_copy, const float* p_gen, const float* schema_ids, float* p_final, int N, int T_q, int vocab_size, int T_k) {
    int total = N * T_q * vocab_size;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointer_blend_forward_kernel<<<numBlocks, BLOCK_SIZE>>>(p_vocab, p_copy, p_gen, schema_ids, p_final, N, T_q, vocab_size, T_k);
}

__global__ void pointer_blend_backward_kernel(const float* d_final, const float* p_vocab, const float* p_copy, const float* p_gen, const float* schema_ids, float* d_vocab, float* d_copy, float* d_pgen, int N, int T_q, int vocab_size, int T_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * T_q;
    if (idx < total) {
        int n = idx / T_q;
        // tq not used but idx is n * T_q + tq
        
        float gen_prob = p_gen[idx];
        float grad_pgen = 0.0f;
        
        // 1. Gradients w.r.t p_vocab
        for (int v = 0; v < vocab_size; ++v) {
            int v_idx = idx * vocab_size + v;
            float grad_out = d_final[v_idx];
            d_vocab[v_idx] = grad_out * gen_prob;
            
            // accumulation for dp_gen
            float copy_prob_v = 0.0f;
            for (int k = 0; k < T_k; ++k) {
                if ((int)schema_ids[n * T_k + k] == v) {
                    copy_prob_v += p_copy[idx * T_k + k];
                }
            }
            grad_pgen += grad_out * (p_vocab[v_idx] - copy_prob_v);
        }
        d_pgen[idx] = grad_pgen;
        
        // 2. Gradients w.r.t p_copy
        for (int k = 0; k < T_k; ++k) {
            int v_id = (int)schema_ids[n * T_k + k];
            int v_idx = idx * vocab_size + v_id;
            d_copy[idx * T_k + k] = d_final[v_idx] * (1.0f - gen_prob);
        }
    }
}

extern "C" void pointer_blend_backward_cuda(const float* d_final, const float* p_vocab, const float* p_copy, const float* p_gen, const float* schema_ids, float* d_vocab, float* d_copy, float* d_pgen, int N, int T_q, int vocab_size, int T_k) {
    int total = N * T_q;
    int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointer_blend_backward_kernel<<<numBlocks, BLOCK_SIZE>>>(d_final, p_vocab, p_copy, p_gen, schema_ids, d_vocab, d_copy, d_pgen, N, T_q, vocab_size, T_k);
}

__global__ void mask_generator_logits_kernel(float* logits, int batch_seq, int vocab_size, int bpe_vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_seq) {
        for (int i = bpe_vocab_size; i < vocab_size; i++) {
            logits[idx * vocab_size + i] = -1e9f;
        }
    }
}

void mask_generator_logits_cuda(float* logits, int batch_seq, int vocab_size, int bpe_vocab_size) {
    int threads = 256;
    int blocks = (batch_seq + threads - 1) / threads;
    mask_generator_logits_kernel<<<blocks, threads>>>(logits, batch_seq, vocab_size, bpe_vocab_size);
}


// ============================================================================
// Deep-fusion additions (Phase B port of scripts/schema_fusion_pt.py)
// ============================================================================

// CE-on-probability gradient with CALLER-CHOSEN label smoothing. The original
// ce_prob_grad_cuda hardcodes eps=0.1; the PyTorch reference uses none, so the
// parity-exact port needs eps=0.
void ce_prob_grad_eps_cuda(const float* P_final, const float* targets_idx, float* dP, int batch_seq, int V, int valid_tokens, float eps)
{
    int total = batch_seq * V; int th = 256; int bl = (total + th - 1) / th;
    ce_prob_grad_kernel<<<bl, th>>>(P_final, targets_idx, dP, batch_seq, V, valid_tokens, eps);
}

// TRUE gate gradient: dL/dp_gen = sum_j dP_final[j] * (P_vocab[j] - P_mem[j]).
// With eps=0 CE, dP_final is nonzero only at the target column, so the sum
// collapses to the target entry. Replaces the old heuristic kernel that pushed
// p_gen toward 1.0 whenever target < 50000.
__global__ void blend_backward_pgen_exact_kernel(const float* dP_final, const float* P_vocab, const float* P_mem, const float* targets_idx, float* dp_gen, int batch_seq, int V)
{
    int bt = blockIdx.x * blockDim.x + threadIdx.x;
    if (bt < batch_seq) {
        int target = (int)targets_idx[bt];
        if (target < 0) { dp_gen[bt] = 0.0f; return; }
        int i = bt * V + target;
        dp_gen[bt] = dP_final[i] * (P_vocab[i] - P_mem[i]);
    }
}
void blend_backward_pgen_exact_cuda(const float* dP_final, const float* P_vocab, const float* P_mem, const float* targets_idx, float* dp_gen, int batch_seq, int V)
{
    int th = 256; int bl = (batch_seq + th - 1) / th;
    blend_backward_pgen_exact_kernel<<<bl, th>>>(dP_final, P_vocab, P_mem, targets_idx, dp_gen, batch_seq, V);
}

// Monitoring-only NLL identical to PyTorch NLLLoss(log(P_target)): no label
// smoothing, target term only. The general cross_entropy_loss_cuda bakes in
// eps=0.1 AND sums -log over all V classes (each masked/zero entry adds
// ~-log(1e-7)), inflating the logged number and making it incomparable to the
// PyTorch reference. Gradient path is unaffected (uses ce_prob_grad_eps).
__global__ void nll_target_loss_kernel(const float* P, const float* targets_idx, float* loss, int batch_seq, int V, int valid_tokens)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_seq) {
        int tgt = (int)targets_idx[i];
        if (tgt < 0 || tgt >= V) return;
        atomicAdd(loss, -logf(P[i * V + tgt] + 1e-9f) / (float)valid_tokens);
    }
}
float nll_target_loss_cuda(const float* P, const float* targets_idx, float* loss, int batch_seq, int V, int valid_tokens)
{
    cudaMemset(loss, 0, sizeof(float));
    int th = 256, bl = (batch_seq + th - 1) / th;
    nll_target_loss_kernel<<<bl, th>>>(P, targets_idx, loss, batch_seq, V, valid_tokens);
    cudaDeviceSynchronize();
    float h = 0.0f; cudaMemcpy(&h, loss, sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}
