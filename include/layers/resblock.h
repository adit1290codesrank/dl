#pragma once
#include "layer.h"
#include "conv2d.h"
#include "batchnorm2d.h"
#include "activation.h"
#include <memory>
#include <fstream>

class ResBlock:public Layer
{
    private:
        Conv2D c1,c2;
        BatchNorm2D b1,b2;
        Activation a1,a2;

        Tensor cached_input;

    public:
        ResBlock(int c);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& dY, float learning_rate) override;

        void set_mode(bool training) override;

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
};