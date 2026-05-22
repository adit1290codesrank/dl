#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "models/transformers/timeseries.h"
#include "include/utils/climate_loader.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline float denorm(float v,float lo,float hi){return v*(hi-lo)+lo;}

float evaluate(TimeSeriesTransformer& model,const ClimateData& d,const std::vector<float>& X,const std::vector<float>& Y,int n,int eb)
{
    int wf=d.window*d.features,H=d.horizon;
    float sum=0.0f;
    int cnt=0;

    for(int b=0;b<n;b+=eb)
    {
        int bs=std::min(eb,n-b);
        std::vector<float> bX(bs*wf),bP(bs*d.window);

        for(int i=0;i<bs;++i)
        {
            std::copy(X.begin()+(b+i)*wf,X.begin()+(b+i+1)*wf,bX.begin()+i*wf);
            for(int t=0;t<d.window;++t) bP[i*d.window+t]=(float)t;
        }

        Tensor dX=Tensor::upload(bX,{bs,d.window,d.features});
        Tensor dP=Tensor::upload(bP,{bs,d.window});
        std::vector<float> p=model.forward_seq(dX,dP).download();

        for(int i=0;i<bs;++i)
            for(int h=0;h<H;++h)
            {
                sum+=std::abs(denorm(p[i*H+h],d.y_min,d.y_max)-denorm(Y[(b+i)*H+h],d.y_min,d.y_max));
                ++cnt;
            }
    }
    return sum/cnt;
}

void per_horizon_mae(TimeSeriesTransformer& model,const ClimateData& d,const std::vector<float>& X,const std::vector<float>& Y,int n,int eb)
{
    int wf=d.window*d.features,H=d.horizon;
    std::vector<float> sums(H,0.0f);
    std::vector<int> cnts(H,0);

    for(int b=0;b<n;b+=eb)
    {
        int bs=std::min(eb,n-b);
        std::vector<float> bX(bs*wf),bP(bs*d.window);

        for(int i=0;i<bs;++i)
        {
            std::copy(X.begin()+(b+i)*wf,X.begin()+(b+i+1)*wf,bX.begin()+i*wf);
            for(int t=0;t<d.window;++t) bP[i*d.window+t]=(float)t;
        }

        Tensor dX=Tensor::upload(bX,{bs,d.window,d.features});
        Tensor dP=Tensor::upload(bP,{bs,d.window});
        std::vector<float> p=model.forward_seq(dX,dP).download();

        for(int i=0;i<bs;++i)
            for(int h=0;h<H;++h)
            {
                sums[h]+=std::abs(denorm(p[i*H+h],d.y_min,d.y_max)-denorm(Y[(b+i)*H+h],d.y_min,d.y_max));
                cnts[h]++;
            }
    }

    std::cout<<"Per-horizon MAE (degC):"<<std::endl;
    for(int h=0;h<H;++h) std::cout<<"  t+"<<(h+1)<<": "<<sums[h]/cnts[h]<<std::endl;
}

int main()
{
    ClimateData d;
    if(!load_climate("data/climate.bin",d))
    {
        std::cerr<<"Run 'python scripts/prepare_climate.py' first"<<std::endl;
        return 1;
    }

    int epochs=50,bs=64,dim=32,heads=4,eb=256;
    float lr_max=0.0005f,lr_min=0.00001f;
    int W=d.window,F=d.features,H=d.horizon,wf=W*F;

    std::cout<<"\n========================================"<<std::endl;
    std::cout<<"Climate Forecasting (Multi-Step)"<<std::endl;
    std::cout<<"========================================"<<std::endl;
    std::cout<<"W="<<W<<" F="<<F<<" H="<<H<<" dim="<<dim<<" heads="<<heads<<" epochs="<<epochs<<" bs="<<bs<<std::endl;
    std::cout<<"train="<<d.n_train<<" val="<<d.n_val<<" test="<<d.n_test<<std::endl;
    std::cout<<"========================================\n"<<std::endl;

    TimeSeriesTransformer model(W,dim,heads,F,H);
    model.set_mode(true);

    int nb=d.n_train/bs;
    Tensor loss_=Tensor::zeros(1,1);
    Tensor grad=Tensor::zeros(bs,H);

    std::vector<int> idx(d.n_train);
    std::iota(idx.begin(),idx.end(),0);
    std::mt19937 rng(std::random_device{}());

    std::ofstream log("outputs/climate_loss.csv");
    if(log.is_open()) log<<"Epoch,TrainLoss,ValMAE,LR\n";

    auto t0=std::chrono::high_resolution_clock::now();

    for(int e=1;e<=epochs;++e)
    {
        std::shuffle(idx.begin(),idx.end(),rng);
        float lr=lr_min+0.5f*(lr_max-lr_min)*(1.0f+cosf((float)(e-1)/(float)(epochs-1)*(float)M_PI));
        float tot=0.0f;

        for(int b=0;b<nb;++b)
        {
            std::vector<float> bX(bs*wf),bP(bs*W),bY(bs*H);

            for(int i=0;i<bs;++i)
            {
                int id=idx[b*bs+i];
                std::copy(d.X_train.begin()+id*wf,d.X_train.begin()+(id+1)*wf,bX.begin()+i*wf);
                for(int t=0;t<W;++t) bP[i*W+t]=(float)t;
                std::copy(d.Y_train.begin()+id*H,d.Y_train.begin()+(id+1)*H,bY.begin()+i*H);
            }

            Tensor dX=Tensor::upload(bX,{bs,W,F});
            Tensor dP=Tensor::upload(bP,{bs,W});
            Tensor dY=Tensor::upload(bY,{bs,H});

            Tensor pred=model.forward_seq(dX,dP);
            Loss::compute_gradient(pred,dY,grad,LossType::MSE);
            model.backward_seq(grad,lr);
            tot+=Loss::compute_loss(pred,dY,loss_,LossType::MSE);
        }

        float avg=tot/nb;
        float vmae=-1.0f;
        if(e%5==0||e==1||e==epochs)
        {
            model.set_mode(false);
            vmae=evaluate(model,d,d.X_val,d.Y_val,d.n_val,eb);
            model.set_mode(true);
        }

        if(e%5==0||e==1)
        {
            std::cout<<"Epoch "<<e<<"/"<<epochs<<"  Loss: "<<avg<<"  LR: "<<lr;
            if(vmae>=0) std::cout<<"  Val MAE: "<<vmae<<" degC";
            std::cout<<std::endl;
        }

        if(log.is_open())
        {
            log<<e<<","<<avg<<","<<(vmae>=0?vmae:0)<<","<<lr<<"\n";
            log.flush();
        }
    }

    if(log.is_open()) log.close();
    float secs=std::chrono::duration<float>(std::chrono::high_resolution_clock::now()-t0).count();
    std::cout<<"\nDone in "<<secs<<"s"<<std::endl;

    std::cout<<"\nEvaluating on test set..."<<std::endl;
    model.set_mode(false);
    std::cout<<"Test MAE (avg): "<<evaluate(model,d,d.X_test,d.Y_test,d.n_test,eb)<<" degC\n"<<std::endl;
    per_horizon_mae(model,d,d.X_test,d.Y_test,d.n_test,eb);

    std::ofstream out("outputs/climate_results.csv");
    out<<"idx";
    for(int h=0;h<H;++h) out<<",true_t+"<<(h+1)<<",pred_t+"<<(h+1);
    out<<"\n";

    for(int b=0;b<d.n_test;b+=eb)
    {
        int bs_=std::min(eb,d.n_test-b);
        std::vector<float> bX(bs_*wf),bP(bs_*W);

        for(int i=0;i<bs_;++i)
        {
            std::copy(d.X_test.begin()+(b+i)*wf,d.X_test.begin()+(b+i+1)*wf,bX.begin()+i*wf);
            for(int t=0;t<W;++t) bP[i*W+t]=(float)t;
        }

        Tensor dX=Tensor::upload(bX,{bs_,W,F});
        Tensor dP=Tensor::upload(bP,{bs_,W});
        std::vector<float> p=model.forward_seq(dX,dP).download();

        for(int i=0;i<bs_;++i)
        {
            out<<(b+i);
            for(int h=0;h<H;++h) out<<","<<denorm(d.Y_test[(b+i)*H+h],d.y_min,d.y_max)<<","<<denorm(p[i*H+h],d.y_min,d.y_max);
            out<<"\n";
        }
    }
    out.close();

    model.save("weights/climate_transformer.bin");
    std::cout<<"\nExported: outputs/climate_results.csv, outputs/climate_loss.csv"<<std::endl;
    return 0;
}
