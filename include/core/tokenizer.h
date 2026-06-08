#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <regex>

class BPETokenizer {
private:
    std::map<std::string, int> w2i;
    std::vector<std::string> i2w;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;
    int unk_id;
    int pad_id;

    std::vector<std::string> get_pairs(const std::vector<std::string>& word) {
        std::vector<std::string> pairs;
        if (word.size() < 2) return pairs;
        for (size_t i = 0; i < word.size() - 1; i++) {
            pairs.push_back(word[i] + " " + word[i+1]);
        }
        return pairs;
    }

public:
    BPETokenizer(const std::string& vocab_path, const std::string& merges_path) {
        std::ifstream vf(vocab_path);
        std::string line;
        int idx = 0;
        while (std::getline(vf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            w2i[line] = idx;
            i2w.push_back(line);
            idx++;
        }
        
        std::ifstream mf(merges_path);
        int rank = 0;
        while (std::getline(mf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            size_t space = line.find(' ');
            if (space != std::string::npos) {
                std::string p1 = line.substr(0, space);
                std::string p2 = line.substr(space + 1);
                bpe_ranks[{p1, p2}] = rank++;
            }
        }
        
        unk_id = w2i.count("[UNK]") ? w2i["[UNK]"] : 1;
        pad_id = w2i.count("[PAD]") ? w2i["[PAD]"] : 0;
    }

    std::vector<std::string> bpe(const std::string& token) {
        std::vector<std::string> word;
        for (char c : token) {
            word.push_back(std::string(1, c));
        }

        while (word.size() > 1) {
            int min_rank = 1e9;
            std::pair<std::string, std::string> best_pair;
            int best_idx = -1;

            for (size_t i = 0; i < word.size() - 1; i++) {
                auto p = std::make_pair(word[i], word[i+1]);
                if (bpe_ranks.count(p)) {
                    if (bpe_ranks[p] < min_rank) {
                        min_rank = bpe_ranks[p];
                        best_pair = p;
                        best_idx = i;
                    }
                }
            }

            if (best_idx == -1) break;

            std::vector<std::string> new_word;
            for (size_t i = 0; i < best_idx; i++) new_word.push_back(word[i]);
            new_word.push_back(word[best_idx] + word[best_idx+1]);
            for (size_t i = best_idx + 2; i < word.size(); i++) new_word.push_back(word[i]);
            
            word = new_word;
        }
        return word;
    }

    std::vector<int> encode(const std::string& text, int max_len = -1) {
        std::vector<int> ids;
        
        // Basic pre-tokenization (mimics HuggingFace Whitespace)
        std::regex re("\\w+|[^\\w\\s]+");
        auto words_begin = std::sregex_iterator(text.begin(), text.end(), re);
        auto words_end = std::sregex_iterator();

        for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
            std::string word = i->str();
            std::vector<std::string> bpe_tokens = bpe(word);
            for (const auto& t : bpe_tokens) {
                if (w2i.count(t)) {
                    ids.push_back(w2i[t]);
                } else {
                    ids.push_back(unk_id);
                }
            }
        }

        if (max_len > 0) {
            if (ids.size() > (size_t)max_len) {
                ids.resize(max_len);
            } else {
                while (ids.size() < (size_t)max_len) {
                    ids.push_back(pad_id);
                }
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string out = "";
        for (int id : ids) {
            if (id == pad_id) break;
            if (id >= 50000) {
                out += "[SCHEMA_" + std::to_string(id - 50000) + "] ";
            } else if (id >= 0 && id < (int)i2w.size()) {
                out += i2w[id];
                // In standard BPE, we might need to handle spaces. 
                // Since this is a simple port, we just append a space 
                // and we can clean up punctuation later.
                out += " "; 
            }
        }
        return out;
    }

    int get_vocab_size() const { return i2w.size(); }
};
