#pragma once
#include "layer.h"
#include "../core/tensor.h"

class Conv2D:public Layer
{
    private:
        int d1,d2,f,p,s;
        Tensor w,b;

        Tensor prev_input,prev_col;

        Tensor mw,vw;
        Tensor mb,vb;
        int t;

        Tensor cached_input;
        Tensor cached_col;
    
    public:
        Conv2D(int d1,int d2,int f,int p,int s);

        Tensor forward(const Tensor& input) override;
        Tensor backward(const Tensor& grad,float learning_rate) override;

        void save(std::ofstream& os) override;
        void load(std::ifstream& is) override;

        int get_c() const {return d2;}
};