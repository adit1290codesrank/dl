#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <memory>
#include <random>
#include <numeric>
#include <algorithm>

#include "models/transformers/timeseries.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline float wave(float x)
{
    return std::sin(x)*std::cos(2.0f*x) + 0.5f*std::sin(3.0f*x);
}

void generate(std::vector<float>& X,std::vector<float>& P,std::vector<float>& Y,int n,int len) 
{
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-2.0f*(float)M_PI, 2.0f*(float)M_PI);

    for(int i=0;i<n;++i) 
    {
        float start_x=dist(gen);
        float step=0.15f; 

        for(int t=0;t<len;++t) 
        {
            float x=start_x+t*step;
            X.push_back(wave(x));
            P.push_back((float)t);
        }
        float next_x=start_x+len*step;
        Y.push_back(wave(next_x));
    }
}

int main() 
{
    int n_samples=10000;
    int epochs=100;
    int batch_size=32;
    float max_lr=0.001f;
    float min_lr=0.00001f;
    int len=10;
    int dim=32;
    int heads=4;
    
    std::cout << "Generating Sequence Data..." << std::endl;
    std::vector<float> X_train,P_train,Y_train;
    generate(X_train,P_train,Y_train,n_samples,len);
    
    TimeSeriesTransformer model(len,dim,heads);
    
    std::cout << "Training Transformer on Sine Wave Sequence Prediction..." << std::endl;
    std::cout << "Config: samples=" << n_samples << " epochs=" << epochs << " batch=" << batch_size 
              << " dim=" << dim << " heads=" << heads << " seq_len=" << len << std::endl;
    std::cout << "========================================" << std::endl;

    model.fit(X_train,P_train,Y_train,n_samples,len,epochs,batch_size,max_lr,min_lr,"outputs/sine_transformer_loss.csv");

    std::cout << "========================================" << std::endl;

    std::cout << "\nEvaluating model..." << std::endl;
    model.set_mode(false);
    
    int test_samples=400;
    std::vector<float> X_test,P_test,Y_test;
    
    for(int i=0;i<test_samples;++i)
    {
        float start_x=-2.0f*(float)M_PI+(4.0f*(float)M_PI*i/test_samples);
        float step=0.15f;
        for(int t=0;t<len;++t)
        {
            X_test.push_back(wave(start_x+t*step));
            P_test.push_back((float)t);
        }
        Y_test.push_back(wave(start_x+len*step));
    }
    
    Tensor d_X_test=Tensor::upload(X_test,{test_samples,len,1});
    Tensor d_P_test=Tensor::upload(P_test,{test_samples,len});
    
    Tensor d_preds=model.forward_seq(d_X_test,d_P_test);
    std::vector<float> h_preds=d_preds.download();
    
    float test_mse=0.0f;
    for(int i=0;i<test_samples;++i)
    {
        float diff=h_preds[i]-Y_test[i];
        test_mse+=diff*diff;
    }
    test_mse/=test_samples;
    std::cout << "Test MSE: " << test_mse << std::endl;
    
    std::ofstream out("outputs/sin_results.csv");
    out << "x_last,y_true_next,y_pred_next\n";
    for(int i=0;i<test_samples;++i) 
    {
        float start_x=-2.0f*(float)M_PI+(4.0f*(float)M_PI*i/test_samples);
        float last_x=start_x+(len-1)*0.15f;
        out << last_x << "," << Y_test[i] << "," << h_preds[i] << "\n";
    }
    out.close();
    
    model.save("weights/sine_transformer.bin");
    std::cout << "Exported predictions to outputs/sin_results.csv" << std::endl;
    return 0;
}
