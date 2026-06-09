#include "../models/transformers/schema_rag_net.h"
#include "../include/core/tokenizer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

void load_breakwalls_dataset(const std::string& path, int& n_train, int& n_val, int& seq_len, int& vocab_size, int& schema_size, int& max_schema_toks,
                             std::vector<float>& X_train, std::vector<float>& Schema_train, std::vector<float>& Y_train,
                             std::vector<float>& X_val, std::vector<float>& Schema_val, std::vector<float>& Y_val,
                             std::vector<float>& schema_vocab_ids)
{
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) throw std::runtime_error("Could not open " + path);

    file.read(reinterpret_cast<char*>(&n_train), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_val), sizeof(int));
    file.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&schema_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&max_schema_toks), sizeof(int));

    // Copy-head targets: one vocab id per schema element (first sub-token).
    schema_vocab_ids.resize(schema_size);
    file.read(reinterpret_cast<char*>(schema_vocab_ids.data()), schema_vocab_ids.size() * sizeof(float));

    int schema_stride = schema_size * max_schema_toks; // sub-tokens per example

    X_train.resize(n_train * seq_len);
    Schema_train.resize(n_train * schema_stride);
    Y_train.resize(n_train * seq_len);

    file.read(reinterpret_cast<char*>(X_train.data()), X_train.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(Schema_train.data()), Schema_train.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(Y_train.data()), Y_train.size() * sizeof(float));

    X_val.resize(n_val * seq_len);
    Schema_val.resize(n_val * schema_stride);
    Y_val.resize(n_val * seq_len);

    file.read(reinterpret_cast<char*>(X_val.data()), X_val.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(Schema_val.data()), Schema_val.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(Y_val.data()), Y_val.size() * sizeof(float));

    file.close();
}

int main(int argc, char** argv)
{
    try {
        // `train_schema_rag resume` fine-tunes from the best checkpoint with a gentle schedule.
        bool resume = (argc > 1 && std::string(argv[1]) == "resume");
        int n_train, n_val, seq_len, vocab_size, schema_size, max_schema_toks;
        std::vector<float> X_train, Schema_train, Y_train, X_val, Schema_val, Y_val, schema_vocab_ids;

        std::cout << "Loading BreakWalls Dataset..." << std::endl;
        load_breakwalls_dataset("data/breakwalls.bin", n_train, n_val, seq_len, vocab_size, schema_size, max_schema_toks,
                                X_train, Schema_train, Y_train, X_val, Schema_val, Y_val, schema_vocab_ids);

        std::cout << "\n========================================" << std::endl;
        std::cout << "Schema-RAG Pointer Network Training" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Seq Len: " << seq_len << " Vocab Size: " << vocab_size << " Schema Size: " << schema_size
                  << " Max Schema Toks: " << max_schema_toks << std::endl;
        std::cout << "Train examples: " << n_train << std::endl;

        int dim = 256;
        int heads = 8;
        int depth = 4;

        std::cout << "Initializing Dual-Encoder Architecture..." << std::endl;
        SchemaRAGNet model(vocab_size, seq_len, dim, heads, depth);

        // Copy-head: map each schema slot to its vocab id so attention can be scattered into the vocab distribution.
        model.set_schema_vocab_ids(Tensor::upload(schema_vocab_ids, {schema_size, 1}));

        // Schedule. Fresh run: peak LR 5e-5 (lowered to stop peak-LR loss spikes).
        int total_epochs = 200;
        float peak_lr = 5e-5f;
        int batch_size = 64;
        int warmup_override = -1; // -1 => fit() uses epochs/10
        if (resume) {
            std::cout << "Resuming from weights/schema_rag.bin (fine-tune mode)..." << std::endl;
            model.load("weights/schema_rag.bin");
            total_epochs = 150;
            peak_lr = 3e-5f;
            warmup_override = 5;
        }
        
        if (n_train < 1000) {
            std::cout << "Detected MINI dataset. Adjusting schedule..." << std::endl;
            total_epochs = 500;
            peak_lr = 5e-5f; // Lowered from 5e-4 to prevent blowout
            batch_size = 8;
        }

        std::cout << "Loading Tokenizer for validation logs..." << std::endl;
        BPETokenizer tokenizer("data/bpe_vocab.txt", "data/bpe_merges.txt");

        std::cout << "Starting Actual Backpropagation Loop..." << std::endl;

        model.fit(X_train, Schema_train, Y_train, X_val, Schema_val, Y_val, n_train, n_val, seq_len, schema_size, max_schema_toks, vocab_size, total_epochs, batch_size, peak_lr, warmup_override, &tokenizer);

        // fit() already checkpoints the best-val model to weights/schema_rag.bin each time it improves,
        // so a run can be stopped at any point. Save the final-epoch weights separately (don't clobber best).
        std::cout << "Saving final-epoch weights to weights/schema_rag_final.bin" << std::endl;
        model.save("weights/schema_rag_final.bin");

        std::cout << "\nModel Training Complete! Best checkpoint is weights/schema_rag.bin" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
