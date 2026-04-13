#include <iostream>
#include <fstream>
#include <random>
#include <numeric>
#include "../../include/core/network.h"
#include "../../include/core/loss.h"
#include <algorithm>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

    float max_lr=learning_rate,min_lr=0.00001f;

    // --- NEW: Open a log file to track loss ---
    std::ofstream log_file("training_loss.csv");
    if (log_file.is_open()) {
        log_file << "Epoch,Loss\n"; // CSV Header
    } else {
        std::cerr << "Warning: Could not open training_loss.csv for writing." << std::endl;
    }

    for(int e=1;e<=epochs;e++)
    {
        std::shuffle(indices.begin(),indices.end(),g);
        float loss=0.0f;
        float current_lr=min_lr+0.5f*(max_lr-min_lr)*(1.0f+cosf((float)(e-1)/(float)(epochs-1)*M_PI));

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

            this->backward(grad,current_lr);
            if(loss_type==LossType::CROSS_ENTROPY) loss+=Loss::compute_loss(output,d_Y,loss_,LossType::CROSS_ENTROPY);
            else if(loss_type==LossType::MSE) loss+=Loss::compute_loss(output,d_Y,loss_,LossType::MSE);
        }
        
        float epoch_loss = loss / num_batches;
        
        if(e%printing_interval==0) {
            std::cout<<"Epoch: "<<e<<", Loss: "<<epoch_loss<<std::endl;
        }

        // --- NEW: Write the current epoch and loss to the file ---
        if (log_file.is_open()) {
            log_file << e << "," << epoch_loss << "\n";
            log_file.flush(); // Crucial: forces write to disk immediately so you don't lose data if it crashes while you sleep
        }
    }
    
    if (log_file.is_open()) {
        log_file.close();
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