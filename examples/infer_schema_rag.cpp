#include "../models/transformers/schema_rag_net.h"
#include "../include/core/tokenizer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <sstream>

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

int main()
{
    try {
        int seq_len, vocab_size, schema_size;
        std::vector<float> Schema;
        
        std::cout << "Loading Database Schema & Metadata..." << std::endl;
        load_metadata_and_schema("data/breakwalls.bin", seq_len, vocab_size, schema_size, Schema);

        std::cout << "Loading BPE Tokenizer..." << std::endl;
        BPETokenizer tokenizer("data/bpe_vocab.txt", "data/bpe_merges.txt");

        int dim = 512; 
        int heads = 8;
        int depth = 6; 

        std::cout << "Initializing Architecture & Loading Weights..." << std::endl;
        SchemaRAGNet model(vocab_size, seq_len, dim, heads, depth);
        
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
            
            std::vector<int> ids = tokenizer.encode(query, seq_len);
            std::vector<float> X(seq_len);
            for(int i = 0; i < seq_len; ++i) X[i] = (float)ids[i];
            
            Tensor dX = Tensor::upload(X, {1, seq_len});
            Tensor dS = Tensor::upload(Schema, {1, schema_size});
            
            Tensor pred = model.forward(dX, dS);
            std::vector<float> out_probs = pred.download();
            
            std::vector<int> out_ids;
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
                out_ids.push_back(best_idx);
            }
            
            std::string sql_out = tokenizer.decode(out_ids);
            // Quick cleanup of typical BPE spacing artifacts for presentation
            std::string clean_sql;
            for(size_t i=0; i<sql_out.length(); i++) {
                if(i > 0 && sql_out[i] == '#' && sql_out[i-1] == '#') continue;
                if(sql_out[i] == '#' && i+1 < sql_out.length() && sql_out[i+1] == '#') {
                    if(!clean_sql.empty() && clean_sql.back() == ' ') clean_sql.pop_back();
                    continue;
                }
                clean_sql += sql_out[i];
            }
            
            std::cout << "SQL  > " << clean_sql << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
