#include "../../include/layers/self_attention.h"
#include <cmath>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>

Tensor matrix_multiply(const Tensor& A,bool transA,const Tensor& B,bool transB);
void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f);

void attention_scale_cuda(float* data,float scale,int size);
void attention_softmax_cuda(float* data,int rows,int cols);
void attention_softmax_backward_cuda(const float* dY,const float* Y,float* dX,int rows,int cols);
void split_heads_cuda(const float* input,float* output,int N,int T,int H,int dk);
void merge_heads_cuda(const float* input,float* output,int N,int T,int H,int dk);
void batched_matmul_cuda(const float* A,bool transA,const float* B,bool transB,float* C,int batch,int M,int K,int N);

SelfAttention::SelfAttention(int dimension,int heads):dimension(dimension),heads(heads),t(0)
{
    if(dimension%heads!=0) throw std::invalid_argument("Dimension must be divisible by number of heads");
    int head_dim=dimension/heads;
    scale=1.0f/sqrtf((float)head_dim);

    std::mt19937 gen(42);
    float stddev=sqrtf(2.0f/dimension);
    std::normal_distribution<float> dist(0.0f,stddev);

    auto init_weight=[&](int r,int c) -> Tensor 
    {
        std::vector<float> h(r*c);
        for(int i=0;i<r*c;i++) h[i]=dist(gen);
        return Tensor::upload(h,r,c);
    };

    wQ=init_weight(dimension,dimension);
    wK=init_weight(dimension,dimension);
    wV=init_weight(dimension,dimension);
    wO=init_weight(dimension,dimension);

    dwQ=Tensor::zeros(dimension,dimension);
    dwK=Tensor::zeros(dimension,dimension);
    dwV=Tensor::zeros(dimension,dimension);
    dwO=Tensor::zeros(dimension,dimension);

    mwq=Tensor::zeros(dimension,dimension); vwq=Tensor::zeros(dimension,dimension);
    mwk=Tensor::zeros(dimension,dimension); vwk=Tensor::zeros(dimension,dimension);
    mwv=Tensor::zeros(dimension,dimension); vwv=Tensor::zeros(dimension,dimension);
    mwo=Tensor::zeros(dimension,dimension); vwo=Tensor::zeros(dimension,dimension);
}

/*
Forward pass:
  Input:  (N*T, D)    where N=batch, T=seq_len, D=dimension
  Output: (N*T, D)

  1. Q = input @ wQ   -> (N*T, D)
  2. K = input @ wK   -> (N*T, D)
  3. V = input @ wV   -> (N*T, D)
  4. Split heads: reshape Q,K,V from (N*T, D) to (N*H, T, dk)
  5. Scores = Q @ K^T  -> (N*H, T, T),  scaled by 1/sqrt(dk)
  6. Attention = softmax(Scores) per row
  7. Context = Attention @ V -> (N*H, T, dk)
  8. Merge heads: (N*H, T, dk) -> (N*T, D)
  9. Output = merged @ wO -> (N*T, D)
*/
Tensor SelfAttention::forward(const Tensor& input)
{
    cached_input=input;

    int D=input.shape.back();      
    int NT=(int)(input.total_elements()/D);
    int H=heads;
    int head_dim=D/H;

    int N=1,T=NT;
    if(input.shape.size()==3)
    {
        N=input.shape[0];
        T=input.shape[1];
    }

    Tensor input_2d=input.reshape({N*T,D});
    Tensor Q=matrix_multiply(input_2d,false,wQ,false);   
    Tensor K=matrix_multiply(input_2d,false,wK,false);
    Tensor V=matrix_multiply(input_2d,false,wV,false);


    Tensor Qh({N*H,T,head_dim});
    Tensor Kh({N*H,T,head_dim});
    Tensor Vh({N*H,T,head_dim});
    split_heads_cuda(Q.data(),Qh.data(),N,T,H,head_dim);
    split_heads_cuda(K.data(),Kh.data(),N,T,H,head_dim);
    split_heads_cuda(V.data(),Vh.data(),N,T,H,head_dim);

    cachedQ=Qh;
    cachedK=Kh;
    cachedV=Vh;


    Tensor scores({N*H,T,T});
    batched_matmul_cuda(Qh.data(),false,Kh.data(),true,scores.data(),N*H,T,head_dim,T);
    attention_scale_cuda(scores.data(),scale,N*H*T*T);

    // 6. Softmax per row: each row of T scores
    Tensor attn({N*H,T,T});
    cudaMemcpy(attn.data(),scores.data(),N*H*T*T*sizeof(float),cudaMemcpyDeviceToDevice);
    attention_softmax_cuda(attn.data(),N*H*T,T);
    cached_attention=attn;

    // 7. Context = Attention @ V -> (N*H, T, dk)
    Tensor context({N*H,T,head_dim});
    batched_matmul_cuda(attn.data(),false,Vh.data(),false,context.data(),N*H,T,T,head_dim);

    // 8. Merge heads: (N*H, T, dk) -> (N*T, D)
    Tensor merged(NT,D);
    merge_heads_cuda(context.data(),merged.data(),N,T,H,head_dim);

    // 9. Output projection
    Tensor output=matrix_multiply(merged,false,wO,false); 
    if(input.shape.size()==3) output.shape={N,T,D};

    return output;
}

/*
Backward pass:
  grad shape: (N*T, D)
  Returns: dInput (N*T, D)
*/
Tensor SelfAttention::backward(const Tensor& dY,float lr)
{
    t++;

    int D=cached_input.shape.back();
    int NT=(int)(cached_input.total_elements()/D);
    int H=heads;
    int head_dim=D/H;

    int N=1,T=NT;
    if(cached_input.shape.size()==3)
    {
        N=cached_input.shape[0];
        T=cached_input.shape[1];
    }

    Tensor dY_flat=dY.reshape({NT,D});

    // 9. Backward through output projection: merged @ wO
    // dMerged = dY @ wO^T,  dwO = merged^T @ dY
    // We need merged — recompute from cached values
    // Context from cached attention and V
    Tensor context({N*H,T,head_dim});
    batched_matmul_cuda(cached_attention.data(),false,cachedV.data(),false,context.data(),N*H,T,T,head_dim);
    Tensor merged(NT,D);
    merge_heads_cuda(context.data(),merged.data(),N,T,H,head_dim);

    Tensor dMerged=matrix_multiply(dY_flat,false,wO,true);     // (N*T, D)
    dwO=matrix_multiply(merged,true,dY_flat,false);            // (D, D)

    // 8. Backward through merge heads (inverse of merge = split)
    Tensor dContext({N*H,T,head_dim});
    split_heads_cuda(dMerged.data(),dContext.data(),N,T,H,head_dim);

    // 7. Backward through Context = Attention @ V
    // dAttn = dContext @ V^T   -> (N*H, T, T)
    // dVh   = Attention^T @ dContext -> (N*H, T, dk)
    Tensor dAttn({N*H,T,T});
    batched_matmul_cuda(dContext.data(),false,cachedV.data(),true,dAttn.data(),N*H,T,head_dim,T);

    Tensor dVh({N*H,T,head_dim});
    batched_matmul_cuda(cached_attention.data(),true,dContext.data(),false,dVh.data(),N*H,T,T,head_dim);

    // 6. Backward through softmax
    Tensor dScores({N*H,T,T});
    attention_softmax_backward_cuda(dAttn.data(),cached_attention.data(),dScores.data(),N*H*T,T);

    // 5. Backward through scale
    attention_scale_cuda(dScores.data(),scale,N*H*T*T);

    // Backward through Scores = Q @ K^T
    // dQh = dScores @ K   -> (N*H, T, dk)
    // dKh = dScores^T @ Q -> (N*H, T, dk)
    Tensor dQh({N*H,T,head_dim});
    batched_matmul_cuda(dScores.data(),false,cachedK.data(),false,dQh.data(),N*H,T,T,head_dim);

    Tensor dKh({N*H,T,head_dim});
    batched_matmul_cuda(dScores.data(),true,cachedQ.data(),false,dKh.data(),N*H,T,T,head_dim);

    // 4. Backward through split heads (inverse of split = merge)
    Tensor dQ(NT,D);
    Tensor dK(NT,D);
    Tensor dV(NT,D);
    merge_heads_cuda(dQh.data(),dQ.data(),N,T,H,head_dim);
    merge_heads_cuda(dKh.data(),dK.data(),N,T,H,head_dim);
    merge_heads_cuda(dVh.data(),dV.data(),N,T,H,head_dim);

    // 1-3. Backward through linear projections
    // dInput = dQ @ wQ^T + dK @ wK^T + dV @ wV^T
    Tensor input_2d=cached_input.reshape({N*T,D});
    dwQ=matrix_multiply(input_2d,true,dQ,false);    // (D, D)
    dwK=matrix_multiply(input_2d,true,dK,false);
    dwV=matrix_multiply(input_2d,true,dV,false);

    Tensor dInput=matrix_multiply(dQ,false,wQ,true);    // (N*T, D)
    Tensor dInput_k=matrix_multiply(dK,false,wK,true);
    Tensor dInput_v=matrix_multiply(dV,false,wV,true);
    dInput=dInput+dInput_k;
    dInput=dInput+dInput_v;

    // Adam updates
    int wsize=dimension*dimension;
    adam_cuda(wQ,dwQ,mwq,vwq,lr,t,wsize);
    adam_cuda(wK,dwK,mwk,vwk,lr,t,wsize);
    adam_cuda(wV,dwV,mwv,vwv,lr,t,wsize);
    adam_cuda(wO,dwO,mwo,vwo,lr,t,wsize);

    if(cached_input.shape.size()==3) dInput.shape={N,T,D};

    return dInput;
}
