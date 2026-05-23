#pragma once
#include "../../include/core/tensor.h"
#include "../../include/layers/layer.h"
#include "../../include/layers/embedding.h"
#include "../../include/layers/transformer.h"
#include "../../include/layers/layernorm.h"
#include "../../include/layers/dense.h"
#include "../../include/layers/activation.h"
#include "../../include/layers/softmax.h"
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

void mean_pool_cuda(const float* seq,float* pool,int batch,int seq_len,int dim);
void mean_pool_backward_cuda(const float* pool,float* seq,int batch,int seq_len,int dim);

/*
BERT for Sequence Classification.

Shape flow (forward):
  Input:           {batch, seq_len}          token IDs
  token_emb:       {batch, seq_len, dim}     embedding lookup appends dim
  pos_emb:         {batch, seq_len, dim}     same
  x = tok + pos:   {batch, seq_len, dim}     3D, this is what Transformer expects
  Transformer[]:   {batch, seq_len, dim}     each block preserves 3D shape
  Mean Pooling:    {batch, dim}              GPU kernel, average across seq_len
  pooler_dense:    {batch, dim}              2D
  classifier:      {batch, num_classes}      2D
  softmax:         {batch, num_classes}      2D
*/

class BERT:public Layer
{
    private:
        int vocab_size,seq_len,dim,depth,num_classes;

        Embedding token_emb;
        Embedding pos_emb;
        Dropout emb_drop;
        std::vector<std::unique_ptr<Transformer>> blocks;

        Dense pooler_dense;
        Activation pooler_act;
        Dropout pooler_drop;
        Dense classifier;
        Softmax sm;

        Tensor cached_enc;

    public:
        BERT(int vocab_size,int seq_len,int dim,int heads,int depth,int num_classes):vocab_size(vocab_size),seq_len(seq_len),dim(dim),depth(depth),num_classes(num_classes),token_emb(vocab_size,dim),pos_emb(seq_len,dim),emb_drop(0.4f),pooler_dense(dim,dim),pooler_act(ActivationType::TANH,0.0f),pooler_drop(0.4f),classifier(dim,num_classes),sm()
        {
            for(int i=0;i<depth;++i) blocks.push_back(std::make_unique<Transformer>(dim,heads,false));
        }

        Tensor forward(const Tensor& X) override
        {
            int batch=X.shape[0];

            Tensor x=token_emb.forward(X);

            std::vector<float> pidx(batch*seq_len);
            for(int b=0;b<batch;++b) for(int t=0;t<seq_len;++t) pidx[b*seq_len+t]=(float)t;
            x=x+pos_emb.forward(Tensor::upload(pidx,{batch,seq_len}));
            x=emb_drop.forward(x);

            for(auto& blk:blocks) x=blk->forward(x);
            cached_enc=x;

            // Mean Pool across seq_len entirely on GPU
            Tensor h(batch,dim);
            mean_pool_cuda(x.data(),h.data(),batch,seq_len,dim);

            h=pooler_dense.forward(h);
            h=pooler_act.forward(h);
            h=pooler_drop.forward(h);
            h=classifier.forward(h);
            return sm.forward(h);
        }

        Tensor backward(const Tensor& dY,float lr) override
        {
            int batch=dY.shape[0];

            Tensor d=sm.backward(dY,lr);
            d=classifier.backward(d,lr);
            d=pooler_drop.backward(d,lr);
            d=pooler_act.backward(d,lr);
            d=pooler_dense.backward(d,lr);

            // Backward Mean Pool entirely on GPU
            Tensor d_enc({batch,seq_len,dim});
            mean_pool_backward_cuda(d.data(),d_enc.data(),batch,seq_len,dim);

            for(int i=depth-1;i>=0;--i) d_enc=blocks[i]->backward(d_enc,lr);

            d_enc=emb_drop.backward(d_enc,lr);

            pos_emb.backward(d_enc,lr);
            token_emb.backward(d_enc,lr);
            return d_enc;
        }

        void set_mode(bool train) override
        {
            token_emb.set_mode(train);
            pos_emb.set_mode(train);
            emb_drop.set_mode(train);
            for(auto& b:blocks) b->set_mode(train);
            pooler_dense.set_mode(train);
            pooler_drop.set_mode(train);
            classifier.set_mode(train);
        }

        void save(const std::string& path)
        {
            std::ofstream os(path,std::ios::binary);
            if(!os.is_open()) throw std::runtime_error("Could not open file for saving: " + path);
            token_emb.save(os);
            pos_emb.save(os);
            for(auto& b:blocks) b->save(os);
            pooler_dense.save(os);
            classifier.save(os);
            os.close();
            std::cout<<"Saved BERT to "<<path<<std::endl;
        }

        void load(const std::string& path)
        {
            std::ifstream is(path,std::ios::binary);
            if(!is.is_open()) throw std::runtime_error("Could not open file for loading: " + path);
            token_emb.load(is);
            pos_emb.load(is);
            for(auto& b:blocks) b->load(is);
            pooler_dense.load(is);
            classifier.load(is);
            is.close();
            std::cout<<"Loaded BERT from "<<path<<std::endl;
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
            int warmup_epochs=std::max(1,std::min(5,epochs/4));

            for(int e=1;e<=epochs;++e)
            {
                std::shuffle(idx.begin(),idx.end(),rng);
                
                float lr;
                if(e<=warmup_epochs) lr=lr_max*((float)e/warmup_epochs);
                else
                {
                    float progress=(float)(e-warmup_epochs)/(float)(epochs-warmup_epochs);
                    lr=lr_min+0.5f*(lr_max-lr_min)*(1.0f+cosf(progress*(float)M_PI));
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

                    Tensor dX=Tensor::upload(bX,{bs,sz});
                    Tensor dY=Tensor::upload(bY,bs,ncls);

                    Tensor pred=forward(dX);
                    Loss::compute_gradient(pred,dY,grad,LossType::CROSS_ENTROPY);
                    backward(grad,lr);
                    tot+=Loss::compute_loss(pred,dY,loss_,LossType::CROSS_ENTROPY);

                    // Zero-overhead progress bar
                    if (b % 5 == 0 || b == nb - 1) {
                        int pct = (b * 100) / nb;
                        std::cout << "\rEpoch " << e << "/" << epochs << " [";
                        for (int p = 0; p < 20; p++) {
                            if (p < pct / 5) std::cout << "=";
                            else if (p == pct / 5) std::cout << ">";
                            else std::cout << " ";
                        }
                        std::cout << "] " << pct << "%  Batch " << b << "/" << nb << std::flush;
                    }
                }

                float avg=tot/nb;
                std::cout<<"\rEpoch "<<e<<"/"<<epochs<<"  Loss: "<<avg<<"  LR: "<<lr<<"                            "<<std::endl;
                if(log.is_open()){log<<e<<","<<avg<<","<<lr<<"\n";log.flush();}
            }

            if(log.is_open()) log.close();
            float secs=std::chrono::duration<float>(std::chrono::high_resolution_clock::now()-t0).count();
            std::cout<<"Done in "<<secs<<"s"<<std::endl;
        }
};
