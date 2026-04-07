#include "../../include/layers/conv2d.h"
#include <cmath>
#include <fstream>
#include <random>

void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size);
void sum_rows_cuda(const Tensor& dY, Tensor& db);
void add_bias_cuda(Tensor& Y, const Tensor& b);
Tensor matrix_multiply(const Tensor& A,bool transA,const Tensor& B,bool transB);

Conv2D::Conv2D(int d1,int d2,int f,int p,int s):d1(d1),d2(d2),f(f),p(p),s(s),t(0)
{
    std::vector<float> h_w(d1*d2*f*f);
    std::vector<float> h_b(d2,0.0f);

    //He initialization
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f,std::sqrt(2.0f/(d1*f*f)));

    for(int i=0;i<d1*d2*f*f;i++) h_w[i]=dist(gen);
    w=Tensor::upload(h_w,d1*f*f,d2);
    b=Tensor::upload(h_b,1,d2);

    mw=Tensor::zeros(d1*f*f,d2);
    vw=Tensor::zeros(d1*f*f,d2);
    mb=Tensor::zeros(1,d2);
    vb=Tensor::zeros(1,d2);
}

Tensor Conv2D::forward(const Tensor& input)
{
    this->cached_input=input;
    this->cached_col=input.im2col(this->f,this->f,this->s,this->p);

    Tensor Y=this->cached_col*this->w;
    add_bias_cuda(Y,this->b);

    int B=this->cached_input.shape[0];
    int OH=(this->cached_input.shape[1]+2*this->p-this->f)/this->s+1;
    int OW=(this->cached_input.shape[2]+2*this->p-this->f)/this->s+1;

    return Y.reshape({B,OH,OW,this->d2});
}

Tensor Conv2D::backward(const Tensor& dY_4d,float learning_rate)
{
    this->t++;
    Tensor dY=dY_4d.reshape({dY_4d.shape[0]*dY_4d.shape[1]*dY_4d.shape[2],this->d2});

    Tensor dW=matrix_multiply(this->cached_col,true,dY,false);

    Tensor db=Tensor::zeros(1,this->d2);
    sum_rows_cuda(dY,db);

    Tensor dX=matrix_multiply(dY,false,this->w,true);
    dX=dX.col2im(this->cached_input.shape,this->f,this->f,this->s,this->p);

    adam_cuda(this->w,dW,this->mw,this->vw,learning_rate,this->t,this->d1*this->f*this->f*this->d2);
    adam_cuda(this->b,db,this->mb,this->vb,learning_rate,this->t,this->d2);
    return dX;
}

void Conv2D::save(std::ofstream& os)
{
    std::vector<float> w_host = this->w.download();
    std::vector<float> b_host = this->b.download();

    os.write(reinterpret_cast<const char*>(w_host.data()), w_host.size() * sizeof(float));
    os.write(reinterpret_cast<const char*>(b_host.data()), b_host.size() * sizeof(float));
}

void Conv2D::load(std::ifstream& is) 
{
    int weight_rows = this->d1 * this->f * this->f;
    std::vector<float> w_host(weight_rows * this->d2);
    std::vector<float> b_host(this->d2);

    is.read(reinterpret_cast<char*>(w_host.data()), w_host.size() * sizeof(float));
    is.read(reinterpret_cast<char*>(b_host.data()), b_host.size() * sizeof(float));

    this->w = Tensor::upload(w_host, weight_rows, this->d2);
    this->b = Tensor::upload(b_host, 1, this->d2);
}
