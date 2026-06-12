#include "../models/transformers/deep_fusion_net.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

// fusion.bin layout (written by scripts/prepare_fusion.py):
//   header: 9 x int32  -- n_train, n_val, seq_len, V, V_bpe, M, max_mem_toks, S, J
//   mem_emit_ids [M]          float
//   mem_tokens   [M * mt]     float
//   mem_types    [M]          float   (0 table / 1 column / 2 fragment)
//   X_train [n_train*seq_len], Y_train [n_train*seq_len]
//   X_val   [n_val*seq_len],   Y_val   [n_val*seq_len]
// The memory bank is stored ONCE (it is global), unlike breakwalls.bin which
// duplicated the schema matrix per example.
void load_fusion_dataset(const std::string& path,
                         int& n_train, int& n_val, int& seq_len, int& V, int& V_bpe,
                         int& M, int& max_mem_toks, int& S, int& J,
                         std::vector<float>& mem_emit_ids, std::vector<float>& mem_tokens,
                         std::vector<float>& mem_types,
                         std::vector<float>& X_train, std::vector<float>& Y_train,
                         std::vector<float>& X_val, std::vector<float>& Y_val)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Could not open " + path);

    int hdr[9];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    n_train = hdr[0]; n_val = hdr[1]; seq_len = hdr[2]; V = hdr[3]; V_bpe = hdr[4];
    M = hdr[5]; max_mem_toks = hdr[6]; S = hdr[7]; J = hdr[8];

    mem_emit_ids.resize(M);
    f.read(reinterpret_cast<char*>(mem_emit_ids.data()), M * sizeof(float));
    mem_tokens.resize((size_t)M * max_mem_toks);
    f.read(reinterpret_cast<char*>(mem_tokens.data()), mem_tokens.size() * sizeof(float));
    mem_types.resize(M);
    f.read(reinterpret_cast<char*>(mem_types.data()), M * sizeof(float));

    X_train.resize((size_t)n_train * seq_len);
    Y_train.resize((size_t)n_train * seq_len);
    f.read(reinterpret_cast<char*>(X_train.data()), X_train.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(Y_train.data()), Y_train.size() * sizeof(float));

    X_val.resize((size_t)n_val * seq_len);
    Y_val.resize((size_t)n_val * seq_len);
    f.read(reinterpret_cast<char*>(X_val.data()), X_val.size() * sizeof(float));
    f.read(reinterpret_cast<char*>(Y_val.data()), Y_val.size() * sizeof(float));

    f.close();
}

int main(int argc, char** argv)
{
    try {
        int n_train, n_val, seq_len, V, V_bpe, M, max_mem_toks, S, J;
        std::vector<float> mem_emit_ids, mem_tokens, mem_types, X_train, Y_train, X_val, Y_val;

        std::cout << "Loading fusion dataset..." << std::endl;
        load_fusion_dataset("data/fusion.bin", n_train, n_val, seq_len, V, V_bpe,
                            M, max_mem_toks, S, J,
                            mem_emit_ids, mem_tokens, mem_types,
                            X_train, Y_train, X_val, Y_val);

        std::cout << "\n========================================" << std::endl;
        std::cout << "DeepFusionNet Training (Phase B C++ port)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "train=" << n_train << " val=" << n_val << " seq_len=" << seq_len
                  << " V=" << V << " V_bpe=" << V_bpe
                  << " M=" << M << " mem_toks=" << max_mem_toks
                  << " S=" << S << " J=" << J << std::endl;

        // Frozen Phase A production config (512/8/0.15). Override via argv for
        // smoke tests: train_deep_fusion [dim] [depth] [epochs] [bs]
        int dim    = (argc > 1) ? std::stoi(argv[1]) : 512;
        int depth  = (argc > 2) ? std::stoi(argv[2]) : 8;
        int epochs = (argc > 3) ? std::stoi(argv[3]) : 40;
        int bs     = (argc > 4) ? std::stoi(argv[4]) : 64;
        int heads = 8;
        float dropout = 0.15f;
        float peak_lr = 2.5e-4f;
        int warmup = 10;

        std::cout << "Model: dim=" << dim << " heads=" << heads << " depth=" << depth
                  << " dropout=" << dropout << std::endl;

        DeepFusionNet model(V, V_bpe, seq_len, dim, heads, depth, dropout);
        model.set_memory(Tensor::upload(mem_tokens, {M, max_mem_toks}),
                         Tensor::upload(mem_types, {1, M}),
                         Tensor::upload(mem_emit_ids, {M, 1}));

        model.fit(X_train, Y_train, X_val, Y_val, n_train, n_val, seq_len,
                  epochs, bs, peak_lr, warmup);

        std::cout << "\nTraining complete. Best checkpoint: weights/deep_fusion.bin" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
