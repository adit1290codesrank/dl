#pragma once
#include "../../include/core/tensor.h"
#include "../../include/layers/layer.h"
#include "../../include/layers/conv2d.h"
#include "../../include/layers/embedding.h"
#include "../../include/layers/transformer.h"
#include "../../include/layers/dense.h"
#include "../../include/layers/activation.h"
#include "../../include/layers/softmax.h"
#include "../../include/layers/augment.h"
#include "../../include/layers/dropout.h"
#include "../../include/core/loss.h"
#include <memory>
#include <vector>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class VisionTransformer:public Layer
{
    private:
        int img,P,G,N,C,dim,depth,cls;

        Augment aug;
        Conv2D patch_conv;
        Embedding pos;
        std::vector<std::unique_ptr<Transformer>> blocks;
        Dense head1;
        Activation act;
        Dropout drop;
        Dense head2;
        Softmax sm;

        Tensor cached_enc;

    public:
        VisionTransformer(int img,int patch,int dim,int heads,int depth,int cls,int aug_pad=4,int aug_cut=8):img(img),P(patch),G(img/patch),N((img/patch)*(img/patch)),C(3),dim(dim),depth(depth),cls(cls),aug(img,img,3,aug_pad,aug_cut),patch_conv(3,dim,patch,0,patch),pos(N,dim),head1(N*dim,dim),act(ActivationType::LEAKY_RELU,0.01f),drop(0.1f),head2(dim,cls),sm()
        {
            for(int i=0;i<depth;++i) blocks.push_back(std::make_unique<Transformer>(dim,heads,false));
        }

        Tensor forward(const Tensor& X) override
        {
            int batch=X.shape[0];
            Tensor x=X.reshape({batch,img,img,C});
            x=aug.forward(x);
            x=patch_conv.forward(x);
            x=x.reshape({batch,N,dim});

            std::vector<float> pidx(batch*N);
            for(int b=0;b<batch;++b) for(int t=0;t<N;++t) pidx[b*N+t]=(float)t;
            x=x+pos.forward(Tensor::upload(pidx,{batch,N}));

            for(auto& blk:blocks) x=blk->forward(x);

            cached_enc=x;
            x=x.reshape({batch,N*dim});
            x=head1.forward(x);
            x=act.forward(x);
            x=drop.forward(x);
            x=head2.forward(x);
            return sm.forward(x);
        }

        Tensor backward(const Tensor& dY,float lr) override
        {
            int batch=cached_enc.shape[0];
            Tensor d=sm.backward(dY,lr);
            d=head2.backward(d,lr);
            d=drop.backward(d,lr);
            d=act.backward(d,lr);
            d=head1.backward(d,lr);

            d=d.reshape({batch,N,dim});
            for(int i=depth-1;i>=0;--i) d=blocks[i]->backward(d,lr);

            pos.backward(d,lr);

            d=d.reshape({batch,G,G,dim});
            d=patch_conv.backward(d,lr);
            d=aug.backward(d,lr);
            return d.reshape({batch,img*img*C});
        }

        void set_mode(bool train) override
        {
            aug.set_mode(train);
            patch_conv.set_mode(train);
            pos.set_mode(train);
            for(auto& b:blocks) b->set_mode(train);
            head1.set_mode(train);
            drop.set_mode(train);
            head2.set_mode(train);
        }

        void save(const std::string& path)
        {
            std::ofstream os(path,std::ios::binary);
            patch_conv.save(os);
            pos.save(os);
            for(auto& b:blocks) b->save(os);
            head1.save(os);
            head2.save(os);
            os.close();
            std::cout<<"Saved ViT to "<<path<<std::endl;
        }

        void load(const std::string& path)
        {
            std::ifstream is(path,std::ios::binary);
            patch_conv.load(is);
            pos.load(is);
            for(auto& b:blocks) b->load(is);
            head1.load(is);
            head2.load(is);
            is.close();
            std::cout<<"Loaded ViT from "<<path<<std::endl;
        }

        void fit(const std::vector<float>& X,const std::vector<float>& Y,int n,int sz,int ncls,int epochs,int bs,float lr_max,float lr_min=0.00001f,const std::string& csv="")
        {
            set_mode(true);
            int nb=n/bs;
            Tensor loss_=Tensor::zeros(1,1);
            Tensor grad=Tensor::zeros(bs,ncls);

            std::vector<int> idx(n);
            std::iota(idx.begin(),idx.end(),0);
            std::mt19937 rng(std::random_device{}());

            std::ofstream log;
            if(!csv.empty()){log.open(csv);if(log.is_open()) log<<"Epoch,Loss,LR\n";}

            auto t0=std::chrono::high_resolution_clock::now();

            // Calculate warmup epochs (10 epochs or 1/8th of total, whichever is smaller)
            int warmup_epochs = std::max(1, std::min(10, epochs / 8));

            for(int e=1;e<=epochs;++e)
            {
                std::shuffle(idx.begin(),idx.end(),rng);
                
                // Linear Warmup + Cosine Decay
                float lr;
                if (e <= warmup_epochs) {
                    lr = lr_max * ((float)e / warmup_epochs);
                } else {
                    float progress = (float)(e - warmup_epochs) / (float)(epochs - warmup_epochs);
                    lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + cosf(progress * (float)M_PI));
                }

                float tot=0.0f;

                for(int b=0;b<nb;++b)
                {
                    std::vector<float> bX(bs*sz),bY(bs*ncls);
                    for(int i=0;i<bs;++i)
                    {
                        int id=idx[b*bs+i];
                        std::copy(X.begin()+id*sz,X.begin()+(id+1)*sz,bX.begin()+i*sz);
                        std::copy(Y.begin()+id*ncls,Y.begin()+(id+1)*ncls,bY.begin()+i*ncls);
                    }

                    Tensor dX=Tensor::upload(bX,bs,sz);
                    Tensor dY=Tensor::upload(bY,bs,ncls);

                    Tensor pred=forward(dX);
                    Loss::compute_gradient(pred,dY,grad,LossType::CROSS_ENTROPY);
                    backward(grad,lr);
                    tot+=Loss::compute_loss(pred,dY,loss_,LossType::CROSS_ENTROPY);
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
