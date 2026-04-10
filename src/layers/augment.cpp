#include "../../include/layers/augment.h"
#include <cuda_runtime.h>
#include <chrono>

void augment_forward_cuda(const Tensor& input, Tensor& output,int* d_tops, int* d_lefts, int* d_flips,int pad, unsigned long long seed);
void augment_backward_cuda(const Tensor& dout, Tensor& din,const int* d_tops, const int* d_lefts, const int* d_flips,int pad);
void cutout_cuda(Tensor& X, int* d_cx, int* d_cy,int cut_size, unsigned long long seed);

Augment::Augment(int H, int W, int C, int pad, int cut_size):H(H), W(W), C(C), pad(pad), cut_size(cut_size), is_training(true),d_tops(nullptr), d_lefts(nullptr), d_flips(nullptr),d_cx(nullptr), d_cy(nullptr), allocated_N(0)
{
    auto now = std::chrono::high_resolution_clock::now();
    seed = static_cast<unsigned long long>(now.time_since_epoch().count());
}

Augment::~Augment()
{
    if (d_tops)  cudaFree(d_tops);
    if (d_lefts) cudaFree(d_lefts);
    if (d_flips) cudaFree(d_flips);
    if (d_cx)    cudaFree(d_cx);
    if (d_cy)    cudaFree(d_cy);
}

Tensor Augment::forward(const Tensor& input)
{
    if (!is_training) return input;

    int N = input.shape[0];

    if (N != allocated_N)
    {
        if (d_tops)  cudaFree(d_tops);
        if (d_lefts) cudaFree(d_lefts);
        if (d_flips) cudaFree(d_flips);
        if (d_cx)    cudaFree(d_cx);
        if (d_cy)    cudaFree(d_cy);

        cudaMalloc(&d_tops,  N * sizeof(int));
        cudaMalloc(&d_lefts, N * sizeof(int));
        cudaMalloc(&d_flips, N * sizeof(int));
        cudaMalloc(&d_cx,    N * sizeof(int));
        cudaMalloc(&d_cy,    N * sizeof(int));
        allocated_N = N;
    }

    Tensor output(input.shape);
    augment_forward_cuda(input, output, d_tops, d_lefts, d_flips, pad, seed++);
    cutout_cuda(output, d_cx, d_cy, cut_size, seed++);  
    return output;
}

Tensor Augment::backward(const Tensor& grad, float lr)
{
    if (!is_training) return grad;
    Tensor din(grad.shape);
    augment_backward_cuda(grad, din, d_tops, d_lefts, d_flips, pad);
    return din;
}

void Augment::set_mode(bool training)
{
    is_training = training;
}