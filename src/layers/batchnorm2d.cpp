#include "../../include/layers/batchnorm2d.h"
#include <vector>
#include <fstream>

void compute_batch_stats_cuda(const Tensor& input, Tensor& mean, Tensor& var);
void batchnorm_cuda(const Tensor& x, const Tensor& mean, const Tensor& var, const Tensor& gamma, const Tensor& beta, Tensor& x_hat, Tensor& output, float epsilon);
void update_running_stats_cuda(Tensor& r_mean, Tensor& r_var, const Tensor& b_mean, const Tensor& b_var, float momentum);
void batchnorm_forward_cuda(const Tensor& dY, const Tensor& x_hat, Tensor& dg, Tensor& db); 
void batchnorm_backward_cuda(const Tensor& dY, const Tensor& x_hat, const Tensor& var, const Tensor& gamma, const Tensor& dg, const Tensor& db, Tensor& dX, float epsilon);
void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f);

BatchNorm2D::BatchNorm2D(int features,float e,float m):features(features),e(e),m(m),is_training(true)
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

Tensor BatchNorm2D::forward(const Tensor& input)
{
    int B=input.shape[0];
    int H=input.shape[1];
    int W=input.shape[2];
    int C=input.shape[3];
    int N=B*H*W;

    this->cached_mean=Tensor::zeros(1,C);
    this->cached_var=Tensor::zeros(1,C);

    Tensor reshaped_input=input.reshape({N,C});
    this->cached_input=reshaped_input.clone();
    Tensor output=reshaped_input.clone();

    if(is_training)
    {
        compute_batch_stats_cuda(reshaped_input,this->cached_mean,this->cached_var);
        batchnorm_cuda(reshaped_input,this->cached_mean,this->cached_var,g,b,this->cached_input,output,e);
        update_running_stats_cuda(this->running_mean,this->running_var,this->cached_mean,this->cached_var,m);
    }
    else batchnorm_cuda(reshaped_input,this->running_mean,this->running_var,g,b,this->cached_input,output,e);
    
    return output.reshape({B,H,W,C});
}

Tensor BatchNorm2D::backward(const Tensor& dY_4d,float learning_rate)
{
    this->t++;

    int B=dY_4d.shape[0];
    int H=dY_4d.shape[1];
    int W=dY_4d.shape[2];
    int C=dY_4d.shape[3];
    int N=B*H*W;

    Tensor dY=dY_4d.reshape({N,C});
    Tensor dg=Tensor::zeros(1,features);
    Tensor db=Tensor::zeros(1,features);
    Tensor dX=dY.clone();

    batchnorm_forward_cuda(dY,this->cached_input,dg,db);
    batchnorm_backward_cuda(dY,this->cached_input,this->cached_var,g,dg,db,dX,e);

    adam_cuda(g,dg,mg,vg,learning_rate,t,features,0.0f);
    adam_cuda(b,db,mm,vm,learning_rate,t,features,0.0f);

    return dX.reshape({B,H,W,C});
}

void BatchNorm2D::set_mode(bool training)
{
    this->is_training=training;
}

void BatchNorm2D::save(std::ofstream& os)
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

void BatchNorm2D::load(std::ifstream& is)
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