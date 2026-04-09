#include "../../include/layers/projresblock.h"

ProjResBlock::ProjResBlock(int in_c,int out_c):c1(in_c, out_c,3,1,1),b1(out_c),a1(ActivationType::LEAKY_RELU, 0.01f),c2(out_c, out_c, 3, 1, 1),b2(out_c),a2(ActivationType::LEAKY_RELU,0.01f),c_proj(in_c,out_c,1,0,1),b_proj(out_c){}

Tensor ProjResBlock::forward(const Tensor& input)
{
    this->cached_input=input;

    Tensor X=c1.forward(input);
    X=b1.forward(X);
    X=a1.forward(X);
    X=c2.forward(X);
    X=b2.forward(X);

    Tensor skip=c_proj.forward(input);
    skip=b_proj.forward(skip);

    return a2.forward(X+skip);
}

Tensor ProjResBlock::backward(const Tensor& dY, float lr)
{
    Tensor dX=a2.backward(dY, lr);

    Tensor dSkip=b_proj.backward(dX, lr);
    dSkip=c_proj.backward(dSkip, lr);

    Tensor dMain=b2.backward(dX, lr);
    dMain=c2.backward(dMain, lr);
    dMain=a1.backward(dMain, lr);
    dMain=b1.backward(dMain, lr);
    dMain=c1.backward(dMain, lr);

    return dMain+dSkip;
}

void ProjResBlock::set_mode(bool training)
{
    c1.set_mode(training);b1.set_mode(training); a1.set_mode(training);
    c2.set_mode(training);b2.set_mode(training);a2.set_mode(training);
    c_proj.set_mode(training);b_proj.set_mode(training);
}

void ProjResBlock::save(std::ofstream& os)
{
    c1.save(os);b1.save(os);
    c2.save(os);b2.save(os);
    c_proj.save(os);b_proj.save(os);
}

void ProjResBlock::load(std::ifstream& is)
{
    c1.load(is);b1.load(is);
    c2.load(is);b2.load(is);
    c_proj.load(is);b_proj.load(is);
}