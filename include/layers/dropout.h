#pragma once
#include "layer.h"

class Dropout:public Layer
{
    private:
        float rate;
        float *mask;
        unsigned long long seed;
        bool is_training;

    public:
        Dropout(float rate);
        ~Dropout();
        
        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& dZ, float learning_rate) override;

        void set_mode(bool training) override;
};