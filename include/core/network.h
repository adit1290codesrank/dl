#pragma once
#include <vector>
#include <memory>
#include <string>
#include "tensor.h"
#include "../layers/layer.h"
#include "loss.h"

class Network
{
    private:
        std::vector<std::unique_ptr<Layer>> layers;

    public:
        void add(std::unique_ptr<Layer> layer){layers.push_back(std::move(layer));}

        void train();
        void eval();

        Tensor forward(Tensor x)
        {
            for(auto& layer:layers)x=layer->forward(x);
            return x;
        }

        void backward(const Tensor& grad,float learning_rate)
        {
            Tensor g=grad;
            for(auto it=layers.rbegin();it!=layers.rend();++it)g=(*it)->backward(g,learning_rate);
        }

        void fit(const std::vector<float>& X,const std::vector<float>& Y,int n,int size,int classes,int epochs,int printing_interval,float learning_rate,int batch_size,LossType loss);
        Tensor predict(const Tensor& X);

        void save(const std::string& filename);
        void load(const std::string& filename);

};
