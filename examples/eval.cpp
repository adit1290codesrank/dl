#include "../models/transformers/bert.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <algorithm>

struct AlexaDataset{
    int n_train,n_val,seq_len,vocab_size,n_classes;
    std::vector<float> train_X,train_Y,val_X,val_Y;
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

    d.train_X.resize(d.n_train*d.seq_len);
    d.train_Y.resize(d.n_train*d.n_classes);
    d.val_X.resize(d.n_val*d.seq_len);
    d.val_Y.resize(d.n_val*d.n_classes);

    file.read(reinterpret_cast<char*>(d.train_X.data()),d.train_X.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.train_Y.data()),d.train_Y.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.val_X.data()),d.val_X.size()*sizeof(float));
    file.read(reinterpret_cast<char*>(d.val_Y.data()),d.val_Y.size()*sizeof(float));
    return d;
}

int argmax(const float* v, int len) {
    int best = 0;
    float bval = v[0];
    for(int i=1; i<len; ++i) if(v[i] > bval) { bval = v[i]; best = i; }
    return best;
}

int main()
{
    try {
        std::cout << "Loading Alexa Dataset..." << std::endl;
        AlexaDataset d = load_alexa("data/alexa.bin");
        
        int dim=768, heads=12, depth=12, cls=d.n_classes;
        
        std::cout << "Initializing BERT Model..." << std::endl;
        BERT model(d.vocab_size, d.seq_len, dim, heads, depth, cls);
        
        std::cout << "Loading weights from weights/alexa.bin..." << std::endl;
        model.load("weights/alexa.bin");
        model.set_mode(false); // Eval mode
        
        int correct = 0;
        int total = d.n_val;
        int batch_size = 64; // Small batch size to avoid hitting memory limits on some GPUs
        
        std::cout << "Evaluating on " << total << " validation samples..." << std::endl;
        
        for (int i = 0; i < total; i += batch_size) {
            int current_batch = std::min(batch_size, total - i);
            
            std::vector<float> batch_x(d.val_X.begin() + i * d.seq_len, d.val_X.begin() + (i + current_batch) * d.seq_len);
            Tensor X = Tensor::upload(batch_x, {current_batch, d.seq_len});
            
            Tensor Y = model.forward(X);
            std::vector<float> probs = Y.download();
            
            for (int b = 0; b < current_batch; ++b) {
                int pred = argmax(probs.data() + b * cls, cls);
                int actual = argmax(d.val_Y.data() + (i + b) * cls, cls);
                if (pred == actual) correct++;
            }
            
            std::cout << "\rProgress: " << (i + current_batch) << " / " << total << std::flush;
        }
        
        float acc = (float)correct / total * 100.0f;
        std::cout << "\n\nValidation Accuracy: " << acc << "% (" << correct << "/" << total << ")" << std::endl;
        
    } catch(const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
