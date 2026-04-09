#include "../../include/layers/augment.h"
#include <cuda_runtime.h>
#include <chrono>

void augment_forward_cuda(const Tensor& input, Tensor& output,int* d_tops, int* d_lefts, int* d_flips,int pad, unsigned long long seed);
void augment_backward_cuda(const Tensor& dout, Tensor& din,const int* d_tops, const int* d_lefts, const int* d_flips,int pad);

Augment::Augment(int H, int W, int C, int pad):H(H), W(W), C(C), pad(pad), is_training(true),d_tops(nullptr), d_lefts(nullptr), d_flips(nullptr), allocated_N(0)
{
    auto now=std::chrono::high_resolution_clock::now();
    seed=static_cast<unsigned long long>(now.time_since_epoch().count());
}

Augment::~Augment()
{
    if (d_tops)  cudaFree(d_tops);
    if (d_lefts) cudaFree(d_lefts);
    if (d_flips) cudaFree(d_flips);
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
        cudaMalloc(&d_tops,  N * sizeof(int));
        cudaMalloc(&d_lefts, N * sizeof(int));
        cudaMalloc(&d_flips, N * sizeof(int));
        allocated_N = N;
    }

    Tensor output(input.shape);
    augment_forward_cuda(input, output, d_tops, d_lefts, d_flips, pad, seed++);
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