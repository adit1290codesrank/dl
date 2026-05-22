#include "../models/transformers/bert.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>

struct IMDBDataset{
    int n_train,n_val,seq_len,vocab_size;
    std::vector<float> train_X,train_Y;
    std::vector<float> val_X,val_Y;
};

IMDBDataset load_imdb(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    if(!file.is_open()) throw std::runtime_error("Could not open "+path);

    IMDBDataset d;
    file.read(reinterpret_cast<char*>(&d.n_train),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.n_val),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.seq_len),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.vocab_size),sizeof(int));

    std::cout<<"Loading IMDB Dataset..."<<std::endl;
    std::cout<<"Train examples: "<<d.n_train<<std::endl;
    std::cout<<"Val examples: "<<d.n_val<<std::endl;
    std::cout<<"Max Seq Length: "<<d.seq_len<<std::endl;
    std::cout<<"Vocab Size: "<<d.vocab_size<<std::endl;

    d.train_X.resize(d.n_train*d.seq_len);
    d.train_Y.resize(d.n_train*2);
    d.val_X.resize(d.n_val*d.seq_len);
    d.val_Y.resize(d.n_val*2);

    file.read(reinterpret_cast<char*>(d.train_X.data()),d.train_X.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.train_Y.data()),d.train_Y.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.val_X.data()),d.val_X.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.val_Y.data()),d.val_Y.size()*sizeof(float));
    
    file.close();
    return d;
}

int argmax_bert(const std::vector<float>& v,int off,int len)
{
    int best=0;
    float bval=v[off];
    for(int i=1;i<len;++i) if(v[off+i]>bval){bval=v[off+i];best=i;}
    return best;
}

float eval_bert(BERT& model,const std::vector<float>& X,const std::vector<float>& Y,int n,int seq_len,int ncls,int eb)
{
    int correct=0;
    for(int b=0;b<n;b+=eb)
    {
        int bs=std::min(eb,n-b);
        std::vector<float> bX(bs*seq_len),bY(bs*ncls);
        std::memcpy(bX.data(),&X[b*seq_len],bs*seq_len*sizeof(float));
        std::memcpy(bY.data(),&Y[b*ncls],bs*ncls*sizeof(float));

        Tensor dX=Tensor::upload(bX,{bs,seq_len});
        std::vector<float> p=model.forward(dX).download();
        for(int i=0;i<bs;++i)
            if(argmax_bert(p,i*ncls,ncls)==argmax_bert(bY,i*ncls,ncls)) ++correct;
    }
    return (float)correct/n*100.0f;
}

int main()
{
    try{
        IMDBDataset d=load_imdb("data/imdb.bin");

        int epochs=20,bs=64,dim=128,heads=4,depth=2,cls=2;
        float lr_max=0.0001f;

        std::cout<<"\n========================================"<<std::endl;
        std::cout<<"BERT on IMDB Sentiment Analysis"<<std::endl;
        std::cout<<"========================================"<<std::endl;
        std::cout<<"seq_len="<<d.seq_len<<" dim="<<dim<<" heads="<<heads<<" depth="<<depth<<std::endl;
        std::cout<<"epochs="<<epochs<<" bs="<<bs<<" train="<<d.n_train<<" val="<<d.n_val<<std::endl;
        std::cout<<"========================================\n"<<std::endl;

        BERT model(d.vocab_size,d.seq_len,dim,heads,depth,cls);

        std::cout<<"Starting Training..."<<std::endl;
        model.fit(d.train_X,d.train_Y,d.n_train,d.seq_len,cls,epochs,bs,lr_max,0.00001f,"logs/bert_imdb_log.csv");

        model.save("weights/bert_imdb.bin");

        // Evaluate
        model.set_mode(false);
        float train_acc=eval_bert(model,d.train_X,d.train_Y,d.n_train,d.seq_len,cls,64);
        float val_acc=eval_bert(model,d.val_X,d.val_Y,d.n_val,d.seq_len,cls,64);
        std::cout<<"\nTrain Accuracy: "<<train_acc<<"%"<<std::endl;
        std::cout<<"Val Accuracy: "<<val_acc<<"%"<<std::endl;

    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<std::endl;
        return 1;
    }
    return 0;
}
