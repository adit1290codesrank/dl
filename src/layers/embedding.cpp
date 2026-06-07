#include "../../include/layers/embedding.h"
#include <stdexcept>
#include <random>
#include <atomic>

void embedding_forward_cuda(const float* X,const float* W,float* Y,int tokens,int dimension,int size);
void embedding_backward_cuda(const float* X,const float* dY,float* dW,int tokens,int dimension,int size);
void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f);
void clip_grad_norm_tensor_cuda(Tensor& grad, float max_norm);
Tensor matrix_add(const Tensor& A, const Tensor& B);

Embedding::Embedding(int size,int dimension):size(size),dimension(dimension)
{
    this->t=0;
    std::vector<float> h_w(size*dimension);
    static std::atomic<int> seed_counter(42);
    std::mt19937 gen(seed_counter++);

    float stddev=sqrt(2.0f/dimension);
    std::normal_distribution<float> dist(0.0f,stddev);

    for(int i=0;i<size*dimension;i++) h_w[i]=dist(gen);

    this->w=Tensor::upload(h_w,size,dimension);
    this->dw=Tensor::zeros(size,dimension);

    this->mw=Tensor::zeros(size,dimension);
    this->vw=Tensor::zeros(size,dimension);
}

Tensor Embedding::forward(const Tensor& input)
{
    this->cached_input=input;
    std::vector<int> shape=input.shape;
    shape.push_back(dimension);
    Tensor output(shape);
    embedding_forward_cuda(input.data(),w.data(),output.data(),input.total_elements(),dimension,size);
    return output;
}

Tensor Embedding::backward(const Tensor& dY, float lr)
{
    dw.zero_();
    embedding_backward_cuda(cached_input.data(),dY.data(),dw.data(),cached_input.total_elements(),dimension,size);

    // Fold in a tied output-projection gradient (if any) so w gets one combined Adam step.
    if(has_ext) {
        dw = matrix_add(dw, pending_ext_grad);
        has_ext = false;
    }

    t++;
    clip_grad_norm_tensor_cuda(dw, 1.0f);
    adam_cuda(w,dw,mw,vw,lr,t,w.total_elements());

    return Tensor();
}

void Embedding::add_external_grad(const Tensor& g)
{
    pending_ext_grad = g;
    has_ext = true;
}

void Embedding::save(std::ofstream& os)
{
    std::vector<float> h=w.download();
    os.write(reinterpret_cast<const char*>(h.data()),h.size()*sizeof(float));
}

void Embedding::load(std::ifstream& is)
{
    std::vector<float> h(size*dimension);
    is.read(reinterpret_cast<char*>(h.data()),h.size()*sizeof(float));
    w=Tensor::upload(h,size,dimension);
}