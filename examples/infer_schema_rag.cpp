#include "../models/transformers/schema_rag_net.h"
#include "../include/core/tokenizer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <sstream>

void load_metadata_and_schema(const std::string& path, int& seq_len, int& vocab_size, int& schema_size, int& max_schema_toks,
                              std::vector<float>& Schema, std::vector<float>& schema_vocab_ids)
{
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()) throw std::runtime_error("Could not open " + path);

    int n_train, n_val;
    file.read(reinterpret_cast<char*>(&n_train), sizeof(int));
    file.read(reinterpret_cast<char*>(&n_val), sizeof(int));
    file.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&schema_size), sizeof(int));
    file.read(reinterpret_cast<char*>(&max_schema_toks), sizeof(int));

    schema_vocab_ids.resize(schema_size);
    file.read(reinterpret_cast<char*>(schema_vocab_ids.data()), schema_vocab_ids.size() * sizeof(float));

    int stride = schema_size * max_schema_toks; // sub-tokens per example (schema is global/identical)

    // Skip X_train, then read the first (global) schema row.
    file.seekg((std::streamoff)n_train * seq_len * sizeof(float), std::ios::cur);
    Schema.resize(stride);
    file.read(reinterpret_cast<char*>(Schema.data()), Schema.size() * sizeof(float));

    file.close();
}

void load_txt_lines(const std::string& path, std::vector<std::string>& lines) {
    std::ifstream file(path);
    if(file.is_open()) {
        std::string line;
        while(std::getline(file, line)) lines.push_back(line);
    }
}

void load_jargon(const std::string& path, std::map<std::string, std::string>& dict) {
    std::ifstream file(path);
    if(file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            size_t pos = line.find('|');
            if(pos != std::string::npos) {
                dict[line.substr(0, pos)] = line.substr(pos + 1);
            }
        }
    }
}

int main()
{
    try {
        int seq_len, vocab_size, schema_size, max_schema_toks;
        std::vector<float> Schema, schema_vocab_ids;

        std::cout << "Loading Database Schema & Metadata..." << std::endl;
        load_metadata_and_schema("data/breakwalls.bin", seq_len, vocab_size, schema_size, max_schema_toks, Schema, schema_vocab_ids);

        std::cout << "Loading RAG Context..." << std::endl;
        std::vector<std::string> schema_lower;
        load_txt_lines("data/schema_strings.txt", schema_lower);
        std::map<std::string, std::string> jargon_dict;
        load_jargon("data/jargon_dict.txt", jargon_dict);

        std::cout << "Loading BPE Tokenizer..." << std::endl;
        BPETokenizer tokenizer("data/bpe_vocab.txt", "data/bpe_merges.txt");

        // Must match training (examples/train_schema_rag.cpp).
        int dim = 256;
        int heads = 8;
        int depth = 4;

        std::cout << "Initializing Architecture & Loading Weights..." << std::endl;
        SchemaRAGNet model(vocab_size, seq_len, dim, heads, depth);

        model.set_schema_vocab_ids(Tensor::upload(schema_vocab_ids, {schema_size, 1}));

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
            
            // Jargon Resolution (Stage 1)
            for(const auto& pair : jargon_dict) {
                size_t pos = 0;
                while((pos = query.find(pair.first, pos)) != std::string::npos) {
                    query.replace(pos, pair.first.length(), pair.second);
                    pos += pair.second.length();
                }
            }
            
            std::vector<int> base_ids = tokenizer.encode(query, seq_len);
            
            // Find where padding starts
            int pad_idx = 0;
            for(; pad_idx < seq_len; ++pad_idx) {
                if(base_ids[pad_idx] == 0) break; // 0 is PAD
            }
            
            // Add SEP token to denote end of English prompt
            int sep_id = 1;
            std::vector<int> sep_tok = tokenizer.encode("\n", 1);
            if(!sep_tok.empty() && sep_tok[0] != 0) sep_id = sep_tok[0];
            
            if(pad_idx < seq_len) {
                base_ids[pad_idx] = sep_id;
                pad_idx++;
            }
            
            std::cout << "SQL  > ";
            std::vector<int> out_ids;
            
            // Autoregressive Generation Loop
            for(int step = 0; step < seq_len - pad_idx; ++step) {
                int current_len = pad_idx + step;
                
                std::vector<float> X(seq_len, 0.0f);
                for(int i = 0; i < current_len; ++i) X[i] = (float)base_ids[i];
                
                Tensor dX = Tensor::upload(X, {1, seq_len});
                Tensor dS = Tensor::upload(Schema, {1, schema_size, max_schema_toks});

                Tensor pred = model.forward(dX, dS);
                std::vector<float> out_probs = pred.download();
                
                int t = current_len - 1; // Predict next token based on the last token's output
                int best_idx = 0;
                float best_prob = -1.0f;
                for(int v = 0; v < vocab_size; ++v) {
                    float p = out_probs[(t * vocab_size) + v];
                    if(p > best_prob) {
                        best_prob = p;
                        best_idx = v;
                    }
                }
                
                // [EOS] is special-token id 4 ([PAD,UNK,CLS,SEP,EOS]); stop there or on PAD.
                const int eos_id = 4;
                if(best_idx == 0 || best_idx == eos_id) break;

                out_ids.push_back(best_idx);
                if(current_len < seq_len) {
                    base_ids[current_len] = best_idx;
                }
            }
            
            std::string sql_out = tokenizer.decode(out_ids);
            
            // Clean up BPE artifacts
            std::string clean_sql;
            for(size_t i=0; i<sql_out.length(); i++) {
                if(i > 0 && sql_out[i] == '#' && sql_out[i-1] == '#') continue;
                if(sql_out[i] == '#' && i+1 < sql_out.length() && sql_out[i+1] == '#') {
                    if(!clean_sql.empty() && clean_sql.back() == ' ') clean_sql.pop_back();
                    continue;
                }
                clean_sql += sql_out[i];
            }
            
            std::cout << clean_sql << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
