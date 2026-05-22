#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "models/transformers/vit.h"
#include "include/utils/tinyimagenet_loader.h"
#include "include/core/loss.h"

int argmax(const std::vector<float>& v,int off,int len)
{
    int best=0;
    float bval=v[off];
    for(int i=1;i<len;++i) if(v[off+i]>bval){bval=v[off+i];best=i;}
    return best;
}

float eval_acc(VisionTransformer& model,const TinyImageNetData& d,const std::vector<uint8_t>& X,const std::vector<uint8_t>& Y,int n,int eb)
{
    int px=d.img_size*d.img_size*3,cls=d.num_classes;
    int correct=0;
    for(int b=0;b<n;b+=eb)
    {
        int bs=std::min(eb,n-b);
        std::vector<float> bX(bs*px);
        for(int i=0;i<bs;++i)
            for(int j=0;j<px;++j)
                bX[i*px+j]=X[(b+i)*px+j]/255.0f;

        Tensor dX=Tensor::upload(bX,bs,px);
        std::vector<float> p=model.forward(dX).download();
        for(int i=0;i<bs;++i) if(argmax(p,i*cls,cls)==(int)Y[b+i]) ++correct;
    }
    return (float)correct/n*100.0f;
}

int main()
{
    TinyImageNetData d;
    if(!load_tinyimagenet("data/tinyimagenet.bin",d))
    {
        std::cerr<<"Run 'python scripts/prepare_tinyimagenet.py' first"<<std::endl;
        return 1;
    }

    int epochs=50,bs=64,dim=192,heads=4,depth=6;
    int cls=d.num_classes,px=d.img_size*d.img_size*3;
    float lr_max=0.0005f;

    std::cout<<"\n========================================"<<std::endl;
    std::cout<<"Vision Transformer on Tiny ImageNet"<<std::endl;
    std::cout<<"========================================"<<std::endl;
    std::cout<<"img="<<d.img_size<<" patch=8 N=64 dim="<<dim<<" heads="<<heads<<" depth="<<depth<<std::endl;
    std::cout<<"epochs="<<epochs<<" bs="<<bs<<" train="<<d.n_train<<" val="<<d.n_val<<" classes="<<cls<<std::endl;
    std::cout<<"========================================\n"<<std::endl;

    VisionTransformer model(d.img_size,8,dim,heads,depth,cls,4,16);
    model.set_mode(true);

    int nb=d.n_train/bs;
    Tensor loss_=Tensor::zeros(1,1);
    Tensor grad=Tensor::zeros(bs,cls);

    std::vector<int> idx(d.n_train);
    std::iota(idx.begin(),idx.end(),0);
    std::mt19937 rng(std::random_device{}());

    std::ofstream log("outputs/vit_tinyimagenet_loss.csv");
    if(log.is_open()) log<<"Epoch,Loss,ValAcc,LR\n";

    auto t0=std::chrono::high_resolution_clock::now();

    for(int e=1;e<=epochs;++e)
    {
        std::shuffle(idx.begin(),idx.end(),rng);
        float lr=0.00001f+0.5f*(lr_max-0.00001f)*(1.0f+cosf((float)(e-1)/(float)(epochs-1)*3.14159265f));
        float tot=0.0f;

        for(int b=0;b<nb;++b)
        {
            std::vector<float> bX(bs*px),bY(bs*cls,0.0f);
            for(int i=0;i<bs;++i)
            {
                int id=idx[b*bs+i];
                for(int j=0;j<px;++j) bX[i*px+j]=d.X_train_raw[id*px+j]/255.0f;
                bY[i*cls+d.Y_train_raw[id]]=1.0f;
            }

            Tensor dX=Tensor::upload(bX,bs,px);
            Tensor dY=Tensor::upload(bY,bs,cls);

            Tensor pred=model.forward(dX);
            Loss::compute_gradient(pred,dY,grad,LossType::CROSS_ENTROPY);
            model.backward(grad,lr);
            tot+=Loss::compute_loss(pred,dY,loss_,LossType::CROSS_ENTROPY);
        }

        float avg=tot/nb;
        float acc=-1.0f;
        if(e%5==0||e==1||e==epochs)
        {
            model.set_mode(false);
            acc=eval_acc(model,d,d.X_val_raw,d.Y_val_raw,d.n_val,128);
            model.set_mode(true);
        }

        if(e%5==0||e==1)
        {
            std::cout<<"Epoch "<<e<<"/"<<epochs<<"  Loss: "<<avg<<"  LR: "<<lr;
            if(acc>=0) std::cout<<"  Val Acc: "<<acc<<"%";
            std::cout<<std::endl;
        }

        if(log.is_open())
        {
            log<<e<<","<<avg<<","<<(acc>=0?acc:0)<<","<<lr<<"\n";
            log.flush();
        }
    }

    if(log.is_open()) log.close();
    float secs=std::chrono::duration<float>(std::chrono::high_resolution_clock::now()-t0).count();

    std::cout<<"\nDone in "<<secs<<"s"<<std::endl;

    model.set_mode(false);
    float final_acc=eval_acc(model,d,d.X_val_raw,d.Y_val_raw,d.n_val,128);
    std::cout<<"Final Val Accuracy: "<<final_acc<<"%"<<std::endl;

    model.save("weights/vit_tinyimagenet.bin");

    return 0;
}
