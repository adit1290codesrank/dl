#include "../../include/layers/transformer.h"
#include <cmath>
#include <random>

void leaky_relu_forward_cuda(Tensor& Y,float alpha);
void add_bias_cuda(Tensor& Y, const Tensor& b);
Tensor matrix_multiply(const Tensor& A,bool transA,const Tensor& B,bool transB);

Transformer::Transformer(int dimension,int heads):dimension(dimension),attention(dimension,heads),ln1(dimension),ln2(dimension),d1(dimension,dimension*4),act(ActivationType::LEAKY_RELU,0.01f),d2(dimension*4,dimension){}

Tensor Transformer::forward(const Tensor& X)
{

    this->cached_input=X;
    
    int n=X.shape[0],t=X.shape[1],d=X.shape[2];

    Tensor output1=this->ln1.forward(X);
    output1=this->attention.forward(output1);           
    output1=output1+X;
    this->cached_output1=output1;

    Tensor norm2=this->ln2.forward(output1).reshape({n*t,d});
    
    Tensor hidden=this->d1.forward(norm2);
    hidden=act.forward(hidden);
    hidden=d2.forward(hidden);
    Tensor output2=hidden.reshape({n,t,d});
    
    Tensor output=output1+output2;
    
    return output;
}

Tensor Transformer::backward(const Tensor& dY,float lr)
{
    int n=cached_input.shape[0],t=cached_input.shape[1],d=cached_input.shape[2];

    Tensor doutput2=dY.reshape({n*t,d});
    Tensor doutput1=dY;

    doutput2=d2.backward(doutput2,lr);
    doutput2=act.backward(doutput2,lr);
    doutput2=d1.backward(doutput2,lr);

    Tensor dnorm2=doutput2.reshape({n,t,d});
    dnorm2=ln2.backward(dnorm2,lr);

    doutput1=doutput1+dnorm2;

    Tensor dattention=attention.backward(doutput1,lr);
    dattention=ln1.backward(dattention,lr);

    Tensor dinput=dattention+doutput1;

    return dinput;
}


