#pragma once
#inlcude "layer.h"
#include "../core/tensor.h"

class Pooling:public Layer
{
    private:
        int size,s;

        int* index;

        std::vector<int> cached_input_shape;
        int cached_input_size;

    public:
        Pooling(int size,int s);
        ~Pooling();

        bool is_training;

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad,float learning_rate) override;

        void set_mode(bool training) override;
}