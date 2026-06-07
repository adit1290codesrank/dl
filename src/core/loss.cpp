#include "../../include/core/loss.h"
#include <stdexcept>
#include <iostream>

void mse_cuda(const Tensor& y_,const Tensor& y,Tensor& dy);
void cross_entropy_cuda(const float* pred, const float* targets_idx, float* dy, int batch_seq, int vocab_size, int valid_tokens);
float mse_loss_cuda(const Tensor& y_,const Tensor& y,Tensor& loss);
float cross_entropy_loss_cuda(const float* pred, const float* targets_idx, float* loss, int batch_seq, int vocab_size, int valid_tokens);

void Loss::compute_gradient(const Tensor& predictions,const Tensor& target,Tensor& grad,int valid_count,LossType type)
{
    if(type==LossType::MSE) {
        if(predictions.shape!=target.shape) throw std::invalid_argument("Shape mismatch for MSE");
        mse_cuda(predictions,target,grad);
    }
    else if(type==LossType::CROSS_ENTROPY) {
        int batch_seq = target.shape[0];
        int vocab_size = predictions.shape.back();
        cross_entropy_cuda(predictions.data(), target.data(), grad.data(), batch_seq, vocab_size, valid_count);
    }
    else throw std::invalid_argument("Unsupported loss type");
}

float Loss::compute_loss(const Tensor& predictions,const Tensor& target,Tensor& loss,int valid_count,LossType type)
{
    if(type==LossType::MSE) {
        if(predictions.shape!=target.shape) throw std::invalid_argument("Shape mismatch for MSE");
        return mse_loss_cuda(predictions,target,loss);
    }
    else if(type==LossType::CROSS_ENTROPY) {
        int batch_seq = target.shape[0];
        int vocab_size = predictions.shape.back();
        return cross_entropy_loss_cuda(predictions.data(), target.data(), loss.data(), batch_seq, vocab_size, valid_count);
    }
    else throw std::invalid_argument("Unsupported loss type");
}




