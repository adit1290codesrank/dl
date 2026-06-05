#include "../models/transformers/bert.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <unordered_map>
#include <sstream>
#include <cctype>

struct AlexaDataset{
    int n_train,n_val,seq_len,vocab_size,n_classes;
    std::vector<float> train_X,train_Y;
    std::vector<float> val_X,val_Y;
};

// C++ Tokenizer exactly matching your friend's python preprocessing
class CorpusTokenizer {
    std::unordered_map<std::string, int> word_to_id;
    int unk_id, pad_id;

    std::string strip_punct(const std::string& word) {
        std::string res;
        for(char c: word) {
            if(std::isalnum(c)) res += std::tolower(c);
        }
        return res;
    }

public:
    CorpusTokenizer(const std::string& vocab_path) {
        std::ifstream f(vocab_path);
        std::string token;
        int id = 0;
        while (std::getline(f, token)) {
            if (!token.empty() && token.back() == '\r') {
                token.pop_back();
            }
            word_to_id[token] = id++;
        }
        unk_id = word_to_id.at("[UNK]");
        pad_id = word_to_id.at("[PAD]");
    }

    std::vector<float> encode(const std::string& text, int max_len = 32) {
        std::vector<float> ids;
        
        std::istringstream iss(text);
        std::string word;
        while (iss >> word && ids.size() < max_len) {
            std::string clean_word = strip_punct(word);
            if(clean_word.empty()) continue;
            auto it = word_to_id.find(clean_word);
            ids.push_back(it != word_to_id.end() ? it->second : unk_id);
        }
        
        while (ids.size() < max_len) ids.push_back(pad_id);
        return ids;
    }
};

AlexaDataset load_alexa(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    if(!file.is_open()) throw std::runtime_error("Could not open "+path);

    AlexaDataset d;
    file.read(reinterpret_cast<char*>(&d.n_train),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.n_val),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.seq_len),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.vocab_size),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.n_classes),sizeof(int));

    std::cout<<"Loading Alexa Dataset..."<<std::endl;
    std::cout<<"Train examples: "<<d.n_train<<std::endl;
    std::cout<<"Val examples: "<<d.n_val<<std::endl;
    std::cout<<"Max Seq Length: "<<d.seq_len<<std::endl;
    std::cout<<"Vocab Size: "<<d.vocab_size<<std::endl;
    std::cout<<"Intents: "<<d.n_classes<<std::endl;

    d.train_X.resize(d.n_train*d.seq_len);
    d.train_Y.resize(d.n_train*d.n_classes);
    d.val_X.resize(d.n_val*d.seq_len);
    d.val_Y.resize(d.n_val*d.n_classes);

    file.read(reinterpret_cast<char*>(d.train_X.data()),d.train_X.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.train_Y.data()),d.train_Y.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.val_X.data()),d.val_X.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.val_Y.data()),d.val_Y.size()*sizeof(float));
    
    file.close();
    return d;
}

int argmax_alexa(const std::vector<float>& v,int off,int len)
{
    int best=0;
    float bval=v[off];
    for(int i=1;i<len;++i) if(v[off+i]>bval){bval=v[off+i];best=i;}
    return best;
}

float eval_alexa(BERT& model,const std::vector<float>& X,const std::vector<float>& Y,int n,int seq_len,int ncls,int eb)
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
            if(argmax_alexa(p,i*ncls,ncls)==argmax_alexa(bY,i*ncls,ncls)) ++correct;
    }
    return (float)correct/n*100.0f;
}

int main()
{
    try{
        AlexaDataset d=load_alexa("data/alexa.bin");

        int epochs=40,bs=256,dim=768,heads=12,depth=12,cls=d.n_classes;
        float lr_max=0.001f;

        std::cout<<"\n========================================"<<std::endl;
        std::cout<<"BERT on Alexa Intent Classification"<<std::endl;
        std::cout<<"========================================"<<std::endl;
        std::cout<<"seq_len="<<d.seq_len<<" dim="<<dim<<" heads="<<heads<<" depth="<<depth<<std::endl;
        std::cout<<"epochs="<<epochs<<" bs="<<bs<<" train="<<d.n_train<<" val="<<d.n_val<<std::endl;
        std::cout<<"========================================\n"<<std::endl;

        BERT model(d.vocab_size,d.seq_len,dim,heads,depth,cls);

        std::cout<<"Starting Training..."<<std::endl;
        model.fit(d.train_X,d.train_Y,d.n_train,d.seq_len,cls,epochs,bs,lr_max,0.00001f,"logs/alexa_log.csv");

        model.save("weights/alexa.bin");

        model.set_mode(false);
        float train_acc=eval_alexa(model,d.train_X,d.train_Y,d.n_train,d.seq_len,cls,64);
        float val_acc=eval_alexa(model,d.val_X,d.val_Y,d.n_val,d.seq_len,cls,64);
        std::cout<<"\nTrain Accuracy: "<<train_acc<<"%"<<std::endl;
        std::cout<<"Val Accuracy: "<<val_acc<<"%"<<std::endl;

        std::cout<<"\nModel Training and Evaluation Complete. Use cli.exe for interactive inference."<<std::endl;


    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<std::endl;
        return 1;
    }
    return 0;
}
