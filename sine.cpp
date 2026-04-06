#include<iostream>
#include<vector>
#include<cmath>
#include<fstream>
#include<memory>
#include"include/core/network.h"
#include"include/layers/dense.h"
#include"include/layers/activation.h"
#include"include/layers/batchnorm1d.h"

#define PI 3.1415926535f

void generate_sin_data(std::vector<float>& X,std::vector<float>& Y,int n)
{
    for(int i=0;i<n;++i)
    {
        float x=-2.0f*PI+(4.0f*PI*i/n);
        X.push_back(x);
        Y.push_back(std::sin(x));
    }
}

int main()
{
    int n_samples=10000,epochs=200,batch_size=32;
    float lr=0.001f;
    
    std::vector<float> X_train,Y_train;
    generate_sin_data(X_train,Y_train,n_samples);
    
    Network net;

    net.add(std::make_unique<Dense>(1,128));
    net.add(std::make_unique<BatchNorm1D>(128));
    net.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU, 0.01f));

    net.add(std::make_unique<Dense>(128,128));
    net.add(std::make_unique<BatchNorm1D>(128));
    net.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU, 0.01f)); 
    net.add(std::make_unique<Dense>(128,1));
    
    std::cout<<"Training on y=sin(x)..."<<std::endl;
    net.fit(X_train,Y_train,n_samples,1,1,epochs,10,lr,batch_size,LossType::MSE);
    
    std::cout<<"\nSaving trained model to sin_model.bin..."<<std::endl;
    net.save("sin_model.bin");
    
    std::cout<<"Creating fresh network and loading weights..."<<std::endl;
    Network test_net;

    test_net.add(std::make_unique<Dense>(1,128));
    test_net.add(std::make_unique<BatchNorm1D>(128));
    test_net.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU, 0.01f)); 

    test_net.add(std::make_unique<Dense>(128,128));
    test_net.add(std::make_unique<BatchNorm1D>(128));
    test_net.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU, 0.01f)); 

    test_net.add(std::make_unique<Dense>(128,1));
    
    test_net.load("sin_model.bin");
    test_net.eval();
    
    int test_points=400;
    std::vector<float> X_test;
    for(int i=0;i<test_points;++i)
    {
        float x=-2.5f*PI+(5.0f*PI*i/test_points);
        X_test.push_back(x);
    }
    
    Tensor d_X_test=Tensor::upload(X_test,test_points,1);
    Tensor d_preds=test_net.predict(d_X_test);
    std::vector<float> h_preds=d_preds.download();
    
    std::ofstream out("sin_results.csv");
    out<<"x,y_true,y_pred\n";
    for(int i=0;i<test_points;++i)
    {
        float x=X_test[i];
        out<<x<<","<<std::sin(x)<<","<<h_preds[i]<<"\n";
    }
    out.close();
    
    std::cout<<"Exported predictions from LOADED model to sin_results.csv"<<std::endl;
    return 0;
}