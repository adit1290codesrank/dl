#pragma once
#include "layer.h"

class Augment : public Layer
{
    private:
        int H, W, C, pad;
        bool is_training;
        unsigned long long seed;

        int* d_tops;
        int* d_lefts;
        int* d_flips;
        int  allocated_N;

    public:
        Augment(int H, int W, int C, int pad = 4);
        ~Augment();

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad, float lr) override;
        void set_mode(bool training) override;
};