#include "../../include/layers/batchnorm1d.h"
#include <vector>
#include <fstream>

void compute_batch_stats_cuda(const Tensor& input, Tensor& mean, Tensor& var);
void batchnorm_cuda(const Tensor& x, const Tensor& mean, const Tensor& var, const Tensor& gamma, const Tensor& beta, Tensor& x_hat, Tensor& output, float epsilon);
void update_running_stats_cuda(Tensor& r_mean, Tensor& r_var, const Tensor& b_mean, const Tensor& b_var, float momentum);
void batchnorm_forward_cuda(const Tensor& dY, const Tensor& x_hat, Tensor& dg, Tensor& db); 
void batchnorm_backward_cuda(const Tensor& dY, const Tensor& x_hat, const Tensor& var, const Tensor& gamma, const Tensor& dg, const Tensor& db, Tensor& dX, float epsilon);
void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f);

BatchNorm1D::BatchNorm1D(int features,float e,float m):features(features),e(e),m(m),is_training(true)
{
    std::vector<float> ones(features,1.0f);

    g=Tensor::upload(ones,1,features);
    b=Tensor::zeros(1,features);

    mg=Tensor::zeros(1,features);
    vg=Tensor::zeros(1,features);
    mm=Tensor::zeros(1,features);
    vm=Tensor::zeros(1,features);
    t=0;

    running_mean=Tensor::zeros(1,features);
    running_var=Tensor::upload(ones,1,features);
}

Tensor BatchNorm1D::forward(const Tensor& input)
{
    this->cached_mean=Tensor::zeros(1,input.cols());
    this->cached_var=Tensor::zeros(1,input.cols());

    this->cached_input=input.clone();
    Tensor output=input.clone();

    if(is_training)
    {
        compute_batch_stats_cuda(input,this->cached_mean,this->cached_var);
        batchnorm_cuda(input,this->cached_mean,this->cached_var,g,b,this->cached_input,output,e);
        update_running_stats_cuda(this->running_mean,this->running_var,this->cached_mean,this->cached_var,m);
    }
    else batchnorm_cuda(input,this->running_mean,this->running_var,g,b,this->cached_input,output,e);
    return output;
}

Tensor BatchNorm1D::backward(const Tensor& grad,float learning_rate)
{
    this->t++;
    Tensor dg=Tensor::zeros(1,features);
    Tensor db=Tensor::zeros(1,features);
    Tensor dX=grad.clone();

    batchnorm_forward_cuda(grad,this->cached_input,dg,db);
    batchnorm_backward_cuda(grad,this->cached_input,this->cached_var,g,dg,db,dX,e);

    adam_cuda(g,dg,mg,vg,learning_rate,t,features,0.0f);
    adam_cuda(b,db,mm,vm,learning_rate,t,features,0.0f);

    return dX;
}

void BatchNorm1D::set_mode(bool training)
{
    this->is_training=training;
}

void BatchNorm1D::save(std::ofstream& os)
{
    std::vector<float> h_g=this->g.download();
    std::vector<float> h_b=this->b.download();

    std::vector<float> h_r_mean=this->running_mean.download();
    std::vector<float> h_r_var=this->running_var.download();

    os.write(reinterpret_cast<const char*>(h_g.data()),h_g.size()*sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b.data()),h_b.size()*sizeof(float));
    os.write(reinterpret_cast<const char*>(h_r_mean.data()),h_r_mean.size()*sizeof(float));
    os.write(reinterpret_cast<const char*>(h_r_var.data()),h_r_var.size()*sizeof(float));
}

void BatchNorm1D::load(std::ifstream& is)
{
    std::vector<float> h_g(this->features);
    std::vector<float> h_b(this->features);
    std::vector<float> h_r_mean(this->features);
    std::vector<float> h_r_var(this->features);

    is.read(reinterpret_cast<char*>(h_g.data()),h_g.size()*sizeof(float));
    is.read(reinterpret_cast<char*>(h_b.data()),h_b.size()*sizeof(float));
    is.read(reinterpret_cast<char*>(h_r_mean.data()),h_r_mean.size()*sizeof(float));
    is.read(reinterpret_cast<char*>(h_r_var.data()),h_r_var.size()*sizeof(float));

    this->g=Tensor::upload(h_g,1,this->features);
    this->b=Tensor::upload(h_b,1,this->features);
    this->running_mean=Tensor::upload(h_r_mean,1,this->features);
    this->running_var=Tensor::upload(h_r_var,1,this->features);
}
