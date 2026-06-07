#include "../../include/core/loss.h"
#include <stdexcept>
#include <iostream>

void mse_cuda(const Tensor& y_,const Tensor& y,Tensor& dy);
float mse_loss_cuda(const Tensor& y_,const Tensor& y,Tensor& loss);

// Integer-index (masked) cross entropy — targets are class ids [batch_seq, 1], -100 = ignore.
void cross_entropy_cuda(const float* pred, const float* targets_idx, float* dy, int batch_seq, int vocab_size, int valid_tokens);
float cross_entropy_loss_cuda(const float* pred, const float* targets_idx, float* loss, int batch_seq, int vocab_size, int valid_tokens);

// Dense one-hot cross entropy (legacy path used by Network/ViT/BERT/timeseries) — targets match predictions shape.
void cross_entropy_dense_cuda(const Tensor& y_, const Tensor& y, Tensor& dy);
float cross_entropy_loss_dense_cuda(const Tensor& y_, const Tensor& y, Tensor& loss);

// Targets are integer class ids when the last dim collapses to 1 while predictions span the vocab.
static bool uses_index_targets(const Tensor& predictions, const Tensor& target)
{
    int vocab_size = predictions.shape.back();
    return target.shape.back() == 1 && vocab_size != 1;
}

void Loss::compute_gradient(const Tensor& predictions,const Tensor& target,Tensor& grad,LossType type,int valid_count)
{
    if(type==LossType::MSE) {
        if(predictions.shape!=target.shape) throw std::invalid_argument("Shape mismatch for MSE");
        mse_cuda(predictions,target,grad);
    }
    else if(type==LossType::CROSS_ENTROPY) {
        if(uses_index_targets(predictions, target)) {
            int batch_seq = target.shape[0];
            int vocab_size = predictions.shape.back();
            int vc = valid_count > 0 ? valid_count : batch_seq;
            cross_entropy_cuda(predictions.data(), target.data(), grad.data(), batch_seq, vocab_size, vc);
        } else {
            if(predictions.shape!=target.shape) throw std::invalid_argument("Shape mismatch for dense cross entropy");
            cross_entropy_dense_cuda(predictions, target, grad);
        }
    }
    else throw std::invalid_argument("Unsupported loss type");
}

float Loss::compute_loss(const Tensor& predictions,const Tensor& target,Tensor& loss,LossType type,int valid_count)
{
    if(type==LossType::MSE) {
        if(predictions.shape!=target.shape) throw std::invalid_argument("Shape mismatch for MSE");
        return mse_loss_cuda(predictions,target,loss);
    }
    else if(type==LossType::CROSS_ENTROPY) {
        if(uses_index_targets(predictions, target)) {
            int batch_seq = target.shape[0];
            int vocab_size = predictions.shape.back();
            int vc = valid_count > 0 ? valid_count : batch_seq;
            return cross_entropy_loss_cuda(predictions.data(), target.data(), loss.data(), batch_seq, vocab_size, vc);
        }
        if(predictions.shape!=target.shape) throw std::invalid_argument("Shape mismatch for dense cross entropy");
        return cross_entropy_loss_dense_cuda(predictions, target, loss);
    }
    else throw std::invalid_argument("Unsupported loss type");
}




