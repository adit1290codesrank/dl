#pragma once
#include "layer.h"
#include <fstream>

class BatchNorm1D:public Layer
{
    private:
        int features;
        float e,m;

        Tensor g,b;

        Tensor mg,vg;
        Tensor mm,vm;
        int t;

        Tensor running_mean,running_var;

        Tensor cached_mean,cached_var;

    public:
        bool is_training;
        
        BatchNorm1D(int features,float e=1e-5f,float m=0.1f);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad,float learning_rate) override;

        void set_mode(bool training) override;

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;
};