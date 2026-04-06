#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include "../../include/core/network.h"
#include "../../include/core/loss.h"
#include <algorithm>

void Network::fit(const std::vector<float>& X,const std::vector<float>& Y,int n,int size,int classes,int epochs,int printing_interval,float learning_rate,int batch_size,LossType loss_type)
{
    this->train();

    std::vector<int> indices(n);
    std::iota(indices.begin(),indices.end(),0);
    std::random_device rd;
    std::mt19937 g(rd());

    int num_batches=n/batch_size;
    Tensor grad=Tensor::zeros(batch_size, classes);
    Tensor loss_=Tensor::zeros(1,1);

    for(int e=1;e<=epochs;e++)
    {
        std::shuffle(indices.begin(),indices.end(),g);
        float loss=0.0f;

        for(int b=0;b<num_batches;b++)
        {
            std::vector<float> batchX(batch_size*size);
            std::vector<float> batchY(batch_size*classes);

            for(int i=0;i<batch_size;i++)
            {
                int idx=indices[b*batch_size+i];
                std::copy(X.begin()+idx*size,X.begin()+(idx+1)*size,batchX.begin()+i*size);
                std::copy(Y.begin()+idx*classes,Y.begin()+(idx+1)*classes,batchY.begin()+i*classes);   
            }

            Tensor d_X=Tensor::upload(batchX,batch_size,size);
            Tensor d_Y=Tensor::upload(batchY,batch_size,classes);

            Tensor output=forward(d_X);

            if(loss_type==LossType::CROSS_ENTROPY) Loss::compute_gradient(output,d_Y,grad,LossType::CROSS_ENTROPY);
            else if(loss_type==LossType::MSE) Loss::compute_gradient(output,d_Y,grad,LossType::MSE);
            else throw std::invalid_argument("Unsupported loss type");

            this->backward(grad,learning_rate);
            if(loss_type==LossType::CROSS_ENTROPY) loss+=Loss::compute_loss(output,d_Y,loss_,LossType::CROSS_ENTROPY);
            else if(loss_type==LossType::MSE) loss+=Loss::compute_loss(output,d_Y,loss_,LossType::MSE);
        }
        if(e%printing_interval==0) std::cout<<"Epoch: "<<e<<", Loss: "<<loss/num_batches<<std::endl;
    }
}

void Network::train()
{
    for(auto& layer:layers) layer->set_mode(true);
}

void Network::eval()
{
    for(auto& layer:layers) layer->set_mode(false);
}

Tensor Network::predict(const Tensor& X)
{
    this->eval();
    Tensor output=X;
    return this->forward(output);
}

void Network::save(const std::string& filename)
{
    std::ofstream os(filename,std::ios::binary);
    if(!os.is_open()) throw std::runtime_error("Failed to open file for saving:" + filename);
    for(auto& layer:layers) layer->save(os);
    os.close();
    std::cout<<"Model saved to "<<filename<<std::endl;
}

void Network::load(const std::string& filename)
{
    std::ifstream is(filename,std::ios::binary);
    if(!is.is_open()) throw std::runtime_error("Failed to open file for loading:" + filename);
    for(auto& layer:layers) layer->load(is);
    is.close();
    std::cout<<"Model loaded from "<<filename<<std::endl;   
}

