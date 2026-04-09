#pragma once
#include "layer.h"
#include "conv2d.h"
#include "batchnorm2d.h"
#include "activation.h"
#include <fstream>

class ProjResBlock:public Layer
{
    private:
        Conv2D c1, c2, c_proj;   
        BatchNorm2D b1, b2, b_proj;
        Activation a1, a2;

        Tensor cached_input;

    public:
        ProjResBlock(int in_c, int out_c);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& dY, float lr) override;

        void set_mode(bool training) override;

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
};