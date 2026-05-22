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
};

AlexaDataset load_metadata(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    if(!file.is_open()) throw std::runtime_error("Could not open "+path);
    AlexaDataset d;
    file.read(reinterpret_cast<char*>(&d.n_train),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.n_val),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.seq_len),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.vocab_size),sizeof(int));
    file.read(reinterpret_cast<char*>(&d.n_classes),sizeof(int));
    file.close();
    return d;
}

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

int argmax_cli(const std::vector<float>& v,int off,int len)
{
    int best=0;
    float bval=v[off];
    for(int i=1;i<len;++i) if(v[off+i]>bval){bval=v[off+i];best=i;}
    return best;
}

int main()
{
    try{
        std::cout << "Loading Model Metadata..." << std::endl;
        AlexaDataset d=load_metadata("data/alexa.bin");

        int dim=768,heads=12,depth=12,cls=d.n_classes;

        std::cout << "Initializing Massive 85M Parameter BERT-Base Model..." << std::endl;
        BERT model(d.vocab_size,d.seq_len,dim,heads,depth,cls);

        std::cout << "Loading Weights from weights/alexa.bin..." << std::endl;
        model.load("weights/alexa.bin");
        model.set_mode(false); // Evaluation mode

        // Load intents for the interactive prompt
        std::vector<std::string> intent_names;
        std::ifstream f_intents("data/intents.txt");
        std::string intent_name;
        while(std::getline(f_intents, intent_name)) {
            if (!intent_name.empty() && intent_name.back() == '\r') {
                intent_name.pop_back();
            }
            if(!intent_name.empty()) intent_names.push_back(intent_name);
        }

        // Initialize tokenizer
        CorpusTokenizer tokenizer("data/vocab.txt");

        std::cout<<"\n========================================"<<std::endl;
        std::cout<<"Interactive Massive 150-Intent Universal AI CLI Ready!"<<std::endl;
        std::cout<<"Type your command (or 'exit' to quit):"<<std::endl;
        std::cout<<"========================================\n"<<std::endl;

        std::string user_input;
        while(true) {
            std::cout<<"\n> ";
            std::getline(std::cin, user_input);
            if(user_input == "exit" || user_input == "quit") break;
            if(user_input.empty()) continue;

            std::vector<float> token_ids = tokenizer.encode(user_input, d.seq_len);
            
            std::cout << "[Tokens]: ";
            int unk_count = 0;
            for (int i = 0; i < 8; i++) {
                int id = (int)token_ids[i];
                if (id == 1) { std::cout << "[UNK] "; unk_count++; }
                else if (id == 0) { std::cout << "[PAD] "; break; }
                else std::cout << id << " ";
            }
            std::cout << "(UNKs in full seq: ";
            for (auto t : token_ids) if ((int)t == 1) unk_count++;
            std::cout << unk_count << "/" << d.seq_len << ")" << std::endl;

            Tensor X = Tensor::upload(token_ids, {1, d.seq_len});
            Tensor Y = model.forward(X);
            std::vector<float> probs = Y.download();

            int best_intent = argmax_cli(probs, 0, cls);
            std::cout<<"[Intent]: "<<intent_names[best_intent]<<std::endl;
        }

    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<std::endl;
        return 1;
    }
    return 0;
}
