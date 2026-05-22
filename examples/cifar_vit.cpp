#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "models/transformers/vit.h"
#include "include/utils/cifar_loader.h"
#include "include/core/loss.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int argmax(const std::vector<float>& v,int off,int len)
{
    int best=0;
    float bval=v[off];
    for(int i=1;i<len;++i) if(v[off+i]>bval){bval=v[off+i];best=i;}
    return best;
}

float eval_acc(VisionTransformer& model,const std::vector<float>& X,const std::vector<float>& Y,int n,int sz,int cls,int eb)
{
    int correct=0;
    for(int b=0;b<n;b+=eb)
    {
        int bs=std::min(eb,n-b);
        std::vector<float> bX(bs*sz),bY(bs*cls);
        for(int i=0;i<bs;++i)
        {
            std::copy(X.begin()+(b+i)*sz,X.begin()+(b+i+1)*sz,bX.begin()+i*sz);
            std::copy(Y.begin()+(b+i)*cls,Y.begin()+(b+i+1)*cls,bY.begin()+i*cls);
        }
        Tensor dX=Tensor::upload(bX,bs,sz);
        std::vector<float> p=model.forward(dX).download();
        for(int i=0;i<bs;++i) if(argmax(p,i*cls,cls)==argmax(bY,i*cls,cls)) ++correct;
    }
    return (float)correct/n*100.0f;
}

int main()
{
    std::vector<float> X_train,Y_train,X_test,Y_test;
    int n_train=0,n_test=0;
    const int cls=10,sz=32*32*3;

    std::vector<std::string> train_files={
        "./data/cifar-10-batches-bin/data_batch_1.bin",
        "./data/cifar-10-batches-bin/data_batch_2.bin",
        "./data/cifar-10-batches-bin/data_batch_3.bin",
        "./data/cifar-10-batches-bin/data_batch_4.bin",
        "./data/cifar-10-batches-bin/data_batch_5.bin",
    };
    std::vector<std::string> test_files={"./data/cifar-10-batches-bin/test_batch.bin"};

    if(!load_cifar10(train_files,X_train,Y_train,n_train,cls)){std::cerr<<"Failed to load train"<<std::endl;return 1;}
    if(!load_cifar10(test_files,X_test,Y_test,n_test,cls)){std::cerr<<"Failed to load test"<<std::endl;return 1;}

    int epochs=80,bs=128,dim=128,heads=4,depth=4;
    float lr_max=0.001f;

    std::cout<<"\n========================================"<<std::endl;
    std::cout<<"Vision Transformer on CIFAR-10"<<std::endl;
    std::cout<<"========================================"<<std::endl;
    std::cout<<"img=32 patch=4 N=64 dim="<<dim<<" heads="<<heads<<" depth="<<depth<<std::endl;
    std::cout<<"epochs="<<epochs<<" bs="<<bs<<" train="<<n_train<<" test="<<n_test<<std::endl;
    std::cout<<"========================================\n"<<std::endl;

    VisionTransformer model(32,4,dim,heads,depth,cls,4,8);

    model.fit(X_train,Y_train,n_train,sz,cls,epochs,bs,lr_max,0.00001f,"outputs/vit_cifar_loss.csv");

    model.set_mode(false);
    float final_acc=eval_acc(model,X_test,Y_test,n_test,sz,cls,256);
    std::cout<<"Final Test Accuracy: "<<final_acc<<"%"<<std::endl;

    model.save("weights/vit_cifar.bin");

    return 0;
}
