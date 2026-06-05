#include "../models/transformers/schema_rag_net.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <sstream>
#include <algorithm>
#include <cctype>

void load_metadata_and_schema(const std::string& path, int& seq_len, int& vocab_size, int& schema_size, std::vector<float>& Schema) 
{
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) throw std::runtime_error("Could not open " + path);

    int n_train, n_val;
    file.read(reinterpret_cast<char*>(&n_train), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_val), sizeof(int));
    file.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&schema_size), sizeof(int));
    
    // We just need one copy of the Schema array for inference
    std::vector<float> X_train(n_train * seq_len);
    file.read(reinterpret_cast<char*>(X_train.data()), X_train.size() * sizeof(float));
    
    std::vector<float> Full_Schema(n_train * schema_size);
    file.read(reinterpret_cast<char*>(Full_Schema.data()), Full_Schema.size() * sizeof(float));
    
    Schema.assign(Full_Schema.begin(), Full_Schema.begin() + schema_size);
    
    file.close();
}

void load_vocab(const std::string& path, std::map<std::string, int>& w2i, std::vector<std::string>& i2w) {
    std::ifstream file(path);
    if(!file.is_open()) throw std::runtime_error("Could not open " + path);
    std::string word;
    int idx = 0;
    while(std::getline(file, word)) {
        if(!word.empty() && word.back() == '\r') word.pop_back();
        w2i[word] = idx++;
        i2w.push_back(word);
    }
}

std::vector<float> tokenize(const std::string& text, const std::map<std::string, int>& w2i, int seq_len) {
    std::string s = text;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    
    std::string spaced = "";
    for(char c : s) {
        if(std::string(".,!?()'\"[]{}").find(c) != std::string::npos) {
            spaced += " "; spaced += c; spaced += " ";
        } else {
            spaced += c;
        }
    }
    
    std::vector<float> toks(seq_len, 0.0f); // Default to PAD (0)
    std::stringstream ss(spaced);
    std::string word;
    int i = 0;
    int unk_id = 1;
    if(w2i.count("[UNK]")) unk_id = w2i.at("[UNK]");
    
    while(ss >> word && i < seq_len) {
        if(w2i.count(word)) toks[i] = w2i.at(word);
        else toks[i] = unk_id;
        i++;
    }
    return toks;
}

int main()
{
    try {
        int seq_len, vocab_size, schema_size;
        std::vector<float> Schema;
        
        std::cout << "Loading Database Schema & Metadata..." << std::endl;
        load_metadata_and_schema("data/breakwalls.bin", seq_len, vocab_size, schema_size, Schema);

        std::map<std::string, int> w2i;
        std::vector<std::string> i2w;
        load_vocab("data/breakwalls_vocab.txt", w2i, i2w);

        int dim = 256; 
        int heads = 8;
        int depth = 4; 

        std::cout << "Initializing Architecture & Loading Weights..." << std::endl;
        int max_len = std::max(seq_len, schema_size);
        SchemaRAGNet model(vocab_size, max_len, dim, heads, depth);
        
        // Disable dropout and load weights
        model.set_mode(false);
        model.load("weights/schema_rag.bin");

        std::cout << "\n========================================" << std::endl;
        std::cout << "Schema-RAG Inference Engine Ready!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        while(true) {
            std::string query;
            std::cout << "\nUSER > ";
            std::getline(std::cin, query);
            if(query.empty() || query == "exit") break;
            
            std::vector<float> X = tokenize(query, w2i, seq_len);
            
            Tensor dX = Tensor::upload(X, {1, seq_len});
            Tensor dS = Tensor::upload(Schema, {1, schema_size});
            
            Tensor pred = model.forward(dX, dS);
            std::vector<float> out_probs = pred.download();
            
            std::cout << "SQL  > ";
            for(int t = 0; t < seq_len; ++t) {
                int best_idx = 0;
                float best_prob = -1.0f;
                for(int v = 0; v < vocab_size; ++v) {
                    float p = out_probs[(t * vocab_size) + v];
                    if(p > best_prob) {
                        best_prob = p;
                        best_idx = v;
                    }
                }
                
                std::string word = i2w[best_idx];
                if(word == "[PAD]") break;
                std::cout << word << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
