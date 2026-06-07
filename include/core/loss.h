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
        static void compute_gradient(const Tensor& predictions,const Tensor& targets,Tensor& grad,LossType type,int valid_count=-1);
        static float compute_loss(const Tensor& predictions,const Tensor& target,Tensor& loss,LossType type,int valid_count=-1);
};

