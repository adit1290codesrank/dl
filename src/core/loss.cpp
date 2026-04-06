#include "../../include/core/loss.h"
#include <stdexcept>
#include <iostream>

void mse_cuda(const Tensor& y_,const Tensor& y,Tensor& dy);
void cross_entropy_cuda(const Tensor& y_, const Tensor& y, Tensor& dy);
float mse_loss_cuda(const Tensor& y_,const Tensor& y,Tensor& loss);
float cross_entropy_loss_cuda(const Tensor& y_, const Tensor& y,Tensor& loss);

void Loss::compute_gradient(const Tensor& predictions,const Tensor& target,Tensor& grad,LossType type)
{
    if(predictions.shape!=target.shape) throw std::invalid_argument("Shape of predictions and targets must match");
    if(type==LossType::MSE) mse_cuda(predictions,target,grad);
    else if(type==LossType::CROSS_ENTROPY) cross_entropy_cuda(predictions,target,grad);
    else throw std::invalid_argument("Unsupported loss type");
}

float Loss::compute_loss(const Tensor& predictions,const Tensor& target,Tensor& loss,LossType type)
{
    if(predictions.shape!=target.shape) throw std::invalid_argument("Shape of predictions and targets must match");
    if(type==LossType::MSE) return mse_loss_cuda(predictions,target,loss);
    else if(type==LossType::CROSS_ENTROPY) return cross_entropy_loss_cuda(predictions,target,loss);
    else throw std::invalid_argument("Unsupported loss type");
}




