#pragma once
#include "../../include/core/tensor.h"
#include "../../include/layers/layer.h"
#include "../../include/layers/embedding.h"
#include "../../include/layers/transformer.h"
#include "../../include/core/loss.h"
#include "../../include/layers/dense.h"
#include <fstream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class TimeSeriesTransformer:public Layer
{
    private:
        int len;
        int dim;
        int features;
        int horizon;

        Dense proj;
        Embedding pos;
        Transformer encoder;
        Dense output;

        Tensor cached_flat;

    public:
        TimeSeriesTransformer(int len,int dim,int heads,int features=1,int horizon=1):len(len),dim(dim),features(features),horizon(horizon),proj(features,dim),pos(len,dim),encoder(dim,heads,false),output(len*dim,horizon){}

        Tensor forward(const Tensor& X) override
        {
            throw std::runtime_error("Use forward_seq instead");
        }

        Tensor backward(const Tensor& dY,float lr) override
        {
            throw std::runtime_error("Use backward_seq instead");
        }

        Tensor forward_seq(const Tensor& X,const Tensor& P)
        {
            int n=X.shape[0];
            Tensor X_flat=X.reshape({n*len,features});
            Tensor X_proj=proj.forward(X_flat);
            X_proj=X_proj.reshape({n,len,dim});

            Tensor P_proj=pos.forward(P);

            Tensor input=X_proj+P_proj;

            Tensor enc_out=encoder.forward(input);

            Tensor out_flat=enc_out.reshape({n,len*dim});
            this->cached_flat=out_flat;

            Tensor Y=output.forward(out_flat);
            return Y;
        }

        void backward_seq(const Tensor& dY,float lr)
        {
            int n=cached_flat.shape[0];

            Tensor d_out=output.backward(dY,lr).reshape({n,len,dim});

            Tensor in=encoder.backward(d_out,lr);

            pos.backward(in,lr);

            Tensor dX=in.reshape({n*len,dim});
            proj.backward(dX,lr);
        }

        void set_mode(bool train) override
        {
            proj.set_mode(train);
            pos.set_mode(train);
            encoder.set_mode(train);
            output.set_mode(train);
        }

        void save(const std::string& path)
        {
            std::ofstream os(path,std::ios::binary);
            proj.save(os);
            pos.save(os);
            encoder.save(os);
            output.save(os);
            os.close();
            std::cout<<"Saved TimeSeriesTransformer to "<<path<<std::endl;
        }

        void load(const std::string& path)
        {
            std::ifstream is(path,std::ios::binary);
            proj.load(is);
            pos.load(is);
            encoder.load(is);
            output.load(is);
            is.close();
            std::cout<<"Loaded TimeSeriesTransformer from "<<path<<std::endl;
        }

        void fit(const std::vector<float>& X,const std::vector<float>& P_data,const std::vector<float>& Y,int n,int wf,int epochs,int bs,float lr_max,float lr_min=0.00001f,const std::string& csv="")
        {
            set_mode(true);
            int nb=n/bs;
            Tensor loss_=Tensor::zeros(1,1);
            Tensor grad=Tensor::zeros(bs,horizon);

            std::vector<int> idx(n);
            std::iota(idx.begin(),idx.end(),0);
            std::mt19937 rng(std::random_device{}());

            std::ofstream log;
            if(!csv.empty()){log.open(csv);if(log.is_open()) log<<"Epoch,Loss,LR\n";}

            auto t0=std::chrono::high_resolution_clock::now();

            for(int e=1;e<=epochs;++e)
            {
                std::shuffle(idx.begin(),idx.end(),rng);
                float lr=lr_min+0.5f*(lr_max-lr_min)*(1.0f+cosf((float)(e-1)/(float)(epochs-1)*(float)M_PI));
                float tot=0.0f;

                for(int b=0;b<nb;++b)
                {
                    std::vector<float> bX(bs*wf),bP(bs*len),bY(bs*horizon);
                    for(int i=0;i<bs;++i)
                    {
                        int id=idx[b*bs+i];
                        std::copy(X.begin()+id*wf,X.begin()+(id+1)*wf,bX.begin()+i*wf);
                        std::copy(P_data.begin()+id*len,P_data.begin()+(id+1)*len,bP.begin()+i*len);
                        std::copy(Y.begin()+id*horizon,Y.begin()+(id+1)*horizon,bY.begin()+i*horizon);
                    }

                    Tensor dX=Tensor::upload(bX,{bs,len,features});
                    Tensor dP=Tensor::upload(bP,{bs,len});
                    Tensor dY=Tensor::upload(bY,bs,horizon);

                    Tensor pred=forward_seq(dX,dP);
                    Loss::compute_gradient(pred,dY,grad,LossType::MSE);
                    backward_seq(grad,lr);
                    tot+=Loss::compute_loss(pred,dY,loss_,LossType::MSE);
                }

                float avg=tot/nb;
                if(e%5==0||e==1||e==epochs) std::cout<<"Epoch "<<e<<"/"<<epochs<<"  Loss: "<<avg<<"  LR: "<<lr<<std::endl;
                if(log.is_open()){log<<e<<","<<avg<<","<<lr<<"\n";log.flush();}
            }

            if(log.is_open()) log.close();
            float secs=std::chrono::duration<float>(std::chrono::high_resolution_clock::now()-t0).count();
            std::cout<<"Done in "<<secs<<"s"<<std::endl;
        }
};