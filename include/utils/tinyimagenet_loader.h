#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

struct TinyImageNetData
{
    int n_train,n_val,img_size,num_classes;
    std::vector<uint8_t> X_train_raw,Y_train_raw;
    std::vector<uint8_t> X_val_raw,Y_val_raw;
};

bool load_tinyimagenet(const std::string& path,TinyImageNetData& d)
{
    std::ifstream f(path,std::ios::binary);
    if(!f.is_open()){std::cerr<<"Failed to open: "<<path<<std::endl;return false;}

    f.read(reinterpret_cast<char*>(&d.n_train),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.n_val),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.img_size),sizeof(int));
    f.read(reinterpret_cast<char*>(&d.num_classes),sizeof(int));

    int px=d.img_size*d.img_size*3;

    d.X_train_raw.resize(d.n_train*px);
    d.Y_train_raw.resize(d.n_train);
    f.read(reinterpret_cast<char*>(d.X_train_raw.data()),d.X_train_raw.size());
    f.read(reinterpret_cast<char*>(d.Y_train_raw.data()),d.Y_train_raw.size());

    d.X_val_raw.resize(d.n_val*px);
    d.Y_val_raw.resize(d.n_val);
    f.read(reinterpret_cast<char*>(d.X_val_raw.data()),d.X_val_raw.size());
    f.read(reinterpret_cast<char*>(d.Y_val_raw.data()),d.Y_val_raw.size());

    if(f.fail()){std::cerr<<"Error reading tinyimagenet binary"<<std::endl;return false;}

    std::cout<<"Loaded TinyImageNet: train="<<d.n_train<<" val="<<d.n_val<<" img="<<d.img_size<<" classes="<<d.num_classes<<std::endl;
    return true;
}
