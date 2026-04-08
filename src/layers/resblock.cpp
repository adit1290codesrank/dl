#include "../../include/layers/resblock.h"

ResBlock::ResBlock(int channels):c1(channels,channels,3,1,1),b1(channels),a1(ActivationType::LEAKY_RELU,0.01f),c2(channels,channels,3,1,1),b2(channels),a2(ActivationType::LEAKY_RELU,0.01f){}

Tensor ResBlock::forward(const Tensor& input)
{
    this->cached_input=input;

    Tensor X=c1.forward(input);
    X=b1.forward(X);
    X=a1.forward(X);

    X=c2.forward(X);
    X=b2.forward(X);

    Tensor output=X+this->cached_input;
    return a2.forward(output);
}

Tensor ResBlock::backward(const Tensor& dY,float learning_rate)
{
    Tensor dX=a2.backward(dY,learning_rate);
    
    Tensor dC=b2.backward(dX,learning_rate);
    dC=c2.backward(dC,learning_rate);
    dC=a1.backward(dC,learning_rate);
    dC=b1.backward(dC,learning_rate);
    dC=c1.backward(dC,learning_rate);

    Tensor dX_=dX+dC;
    return dX_;
}

void ResBlock::set_mode(bool training) 
{
    c1.set_mode(training);
    b1.set_mode(training);
    a1.set_mode(training);
    c2.set_mode(training);
    b2.set_mode(training);
    a2.set_mode(training);
}

void ResBlock::save(std::ofstream& os) 
{
    c1.save(os);
    b1.save(os);
    c2.save(os);
    b2.save(os);
}

void ResBlock::load(std::ifstream& is) 
{
    c1.load(is);
    b1.load(is);
    c2.load(is);
    b2.load(is);
}