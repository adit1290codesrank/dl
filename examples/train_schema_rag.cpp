#include "../models/transformers/schema_rag_net.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>

void load_breakwalls_dataset(const std::string& path, int& n_train, int& n_val, int& seq_len, int& vocab_size, int& schema_size,
                             std::vector<float>& X_train, std::vector<float>& Schema_train, std::vector<float>& Y_train) 
{
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) throw std::runtime_error("Could not open " + path);

    file.read(reinterpret_cast<char*>(&n_train), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_val), sizeof(int));
    file.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&schema_size), sizeof(int));
    
    X_train.resize(n_train * seq_len);
    Schema_train.resize(n_train * schema_size);
    Y_train.resize(n_train * seq_len);

    file.read(reinterpret_cast<char*>(X_train.data()), X_train.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(Schema_train.data()), Schema_train.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(Y_train.data()), Y_train.size() * sizeof(float));
    
    file.close();
}

int main()
{
    try {
        int n_train, n_val, seq_len, vocab_size, schema_size;
        std::vector<float> X_train, Schema_train, Y_train;
        
        std::cout << "Loading BreakWalls Dataset..." << std::endl;
        load_breakwalls_dataset("data/breakwalls.bin", n_train, n_val, seq_len, vocab_size, schema_size, X_train, Schema_train, Y_train);

        std::cout << "\n========================================" << std::endl;
        std::cout << "Schema-RAG Pointer Network Training" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Seq Len: " << seq_len << " Vocab Size: " << vocab_size << " Schema Size: " << schema_size << std::endl;
        std::cout << "Train examples: " << n_train << std::endl;
        
        int dim = 512; // Expanded for 5000+ dataset
        int heads = 8;
        int depth = 6; // Expanded for 5000+ dataset

        std::cout << "Initializing Dual-Encoder Architecture..." << std::endl;
        SchemaRAGNet model(vocab_size, seq_len, dim, heads, depth);

        std::cout << "Starting Actual Backpropagation Loop..." << std::endl;
        
        // Train for 1000 epochs with Cosine Annealing to fully learn the massive dataset
        model.fit(X_train, Schema_train, Y_train, n_train, seq_len, schema_size, vocab_size, 1000, 8, 1e-3f);

        std::cout << "Saving weights to weights/schema_rag.bin" << std::endl;
        model.save("weights/schema_rag.bin");

        std::cout << "\nModel Training Complete! Architecture successfully built." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
