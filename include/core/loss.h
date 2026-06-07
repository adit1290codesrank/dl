#pragma once
#include "tensor.h"

enum class LossType
{
    MSE,
    CROSS_ENTROPY
};

class Loss
{
    public:
        static void compute_gradient(const Tensor& predictions,const Tensor& targets,Tensor& grad,int valid_count,LossType type);
        static float compute_loss(const Tensor& predictions,const Tensor& target,Tensor& loss,int valid_count,LossType type);
};

