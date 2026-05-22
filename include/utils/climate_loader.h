#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

struct ClimateData
{
    int n_train,n_val,n_test,window,features,horizon;
    std::vector<float> X_train,Y_train,X_val,Y_val,X_test,Y_test;
    float y_min,y_max;
};

bool load_climate(const std::string& path,ClimateData& d)
{
    std::ifstream f(path,std::ios::binary);
    if(!f.is_open()){std::cerr<<"Failed to open: "<<path<<std::endl;return false;}

    f.read(reinterpret_cast<char*>(&d.n_train),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.n_val),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.n_test),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.window),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.features),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.horizon),sizeof(int));

    int wf=d.window*d.features;

    d.X_train.resize(d.n_train*wf);d.Y_train.resize(d.n_train*d.horizon);
    f.read(reinterpret_cast<char*>(d.X_train.data()),d.X_train.size()*sizeof(float));
    f.read(reinterpret_cast<char*>(d.Y_train.data()),d.Y_train.size()*sizeof(float));

    d.X_val.resize(d.n_val*wf);d.Y_val.resize(d.n_val*d.horizon);
    f.read(reinterpret_cast<char*>(d.X_val.data()),d.X_val.size()*sizeof(float));
    f.read(reinterpret_cast<char*>(d.Y_val.data()),d.Y_val.size()*sizeof(float));

    d.X_test.resize(d.n_test*wf);d.Y_test.resize(d.n_test*d.horizon);
    f.read(reinterpret_cast<char*>(d.X_test.data()),d.X_test.size()*sizeof(float));
    f.read(reinterpret_cast<char*>(d.Y_test.data()),d.Y_test.size()*sizeof(float));

    f.read(reinterpret_cast<char*>(&d.y_min),sizeof(float));
    f.read(reinterpret_cast<char*>(&d.y_max),sizeof(float));

    if(f.fail()){std::cerr<<"Error reading climate binary"<<std::endl;return false;}

    std::cout<<"Loaded: train="<<d.n_train<<" val="<<d.n_val<<" test="<<d.n_test<<" W="<<d.window<<" F="<<d.features<<" H="<<d.horizon<<" T=["<<d.y_min<<","<<d.y_max<<"]"<<std::endl;
    return true;
}
