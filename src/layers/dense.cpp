#include "../../include/layers/dense.h"
#include <fstream>
#include <cmath>

void adam_cuda(Tensor& W,const Tensor& grad,Tensor& m,Tensor& v,float lr,int t,int size,float lamda=0.001f);
void sum_rows_cuda(const Tensor& dY, Tensor& db);
void add_bias_cuda(Tensor& Y, const Tensor& b);
Tensor matrix_multiply(const Tensor& A,bool transA,const Tensor& B,bool transB);

Dense::Dense(int input_size,int output_size):input_size(input_size),output_size(output_size),t(0)
{
    std::vector<float> h_w(input_size*output_size);
    std::vector<float> h_b(output_size,0.0f);

    //He initialization
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f,std::sqrt(2.0f/input_size));

    for(int i=0;i<input_size*output_size;i++) h_w[i]=dist(gen);

    w=Tensor::upload(h_w,input_size,output_size);
    b=Tensor::upload(h_b,1,output_size);

    mw=Tensor::zeros(input_size,output_size);
    vw=Tensor::zeros(input_size,output_size);
    mb=Tensor::zeros(1,output_size);
    vb=Tensor::zeros(1,output_size);
}

Tensor Dense::forward(const Tensor& input)
{
    this->cached_input=input;
    Tensor Y=input*this->w;
    add_bias_cuda(Y,this->b);
    return Y;
}

Tensor Dense::backward(const Tensor& dY,float learning_rate)
{
    this->t++;
    Tensor dW=matrix_multiply(this->cached_input,true,dY,false);
    Tensor dX=matrix_multiply(dY,false,this->w,true);
    Tensor db=Tensor::zeros(1,this->output_size);
    sum_rows_cuda(dY,db);

    adam_cuda(this->w,dW,this->mw,this->vw,learning_rate,this->t,this->output_size*this->input_size,0.001f);
    adam_cuda(this->b,db,this->mb,this->vb,learning_rate,this->t,this->output_size,0.0f);

    return dX;
}

void Dense::save(std::ofstream& os)
{
    std::vector<float> h_w=this->w.download();
    std::vector<float> h_b=this->b.download();

    os.write(reinterpret_cast<const char*>(h_w.data()),h_w.size()*sizeof(float));
    os.write(reinterpret_cast<const char*>(h_b.data()),h_b.size()*sizeof(float));
}

void Dense::load(std::ifstream& is)
{
    std::vector<float> h_w(this->input_size*this->output_size);
    std::vector<float> h_b(this->output_size);

    is.read(reinterpret_cast<char*>(h_w.data()),h_w.size()*sizeof(float));
    is.read(reinterpret_cast<char*>(h_b.data()),h_b.size()*sizeof(float));

    this->w=Tensor::upload(h_w,this->input_size,this->output_size);
    this->b=Tensor::upload(h_b, 1, this->output_size); 
}

