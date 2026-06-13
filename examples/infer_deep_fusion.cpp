#include "../models/transformers/deep_fusion_net.h"
#include "../include/core/tokenizer.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <map>
#include <regex>
#include <algorithm>
#include <cctype>

// Pure-C++ greedy-decode inference for DeepFusionNet (Phase B).
// Mirrors scripts/schema_fusion_pt.py --ask: delexicalize literal values to
// [valN] slots, greedy-decode, expand atomic ids to exact schema/jargon
// strings, substitute the real values back. No Python at inference time.

// ---- value slots (port of scripts/value_slots.py) -------------------------
static const int DEFAULT_YEAR = 2025;
static const char* MONTHS[12] = {"jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"};

static int month_index(const std::string& m3) {
    for (int i = 0; i < 12; ++i) if (m3 == MONTHS[i]) return i + 1;
    return 0;
}
static int days_in_month(int y, int m) {
    static const int d[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    if (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)) return 29;
    return d[m - 1];
}
static std::string iso(int y, int m, int d) {
    char buf[16]; snprintf(buf, sizeof(buf), "%04d-%02d-%02d", y, m, d); return buf;
}

struct Span { size_t start, end; std::vector<std::string> vals; };

// Returns delexed text (with " [valN] " markers) and the slot value list.
static std::string extract_slots(const std::string& text, std::vector<std::string>& slot_values)
{
    std::vector<Span> spans;
    auto overlaps = [&](size_t s, size_t e) {
        for (auto& sp : spans) if (!(e <= sp.start || s >= sp.end)) return true;
        return false;
    };

    // 1. ISO date  YYYY-M-D
    {
        std::regex re(R"(\b(\d{4})-(\d{1,2})-(\d{1,2})\b)");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
            auto m = *it;
            int y = std::stoi(m[1]), mo = std::stoi(m[2]), d = std::stoi(m[3]);
            spans.push_back({(size_t)m.position(), (size_t)(m.position() + m.length()), {iso(y, mo, d)}});
        }
    }
    // 2. Month + year  -> two slots (month start, month end)
    {
        std::regex re(R"(\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{4})\b)", std::regex::icase);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
            auto m = *it;
            if (overlaps(m.position(), m.position() + m.length())) continue;
            std::string mm = m[1]; std::transform(mm.begin(), mm.end(), mm.begin(), ::tolower);
            int mo = month_index(mm.substr(0, 3)), y = std::stoi(m[2]);
            spans.push_back({(size_t)m.position(), (size_t)(m.position() + m.length()),
                             {iso(y, mo, 1), iso(y, mo, days_in_month(y, mo))}});
        }
    }
    // 3. Month with NO year -> DEFAULT_YEAR (exclude "may"/modal collisions handled by full forms)
    {
        std::regex re(R"(\b(january|february|march|april|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b)", std::regex::icase);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
            auto m = *it;
            if (overlaps(m.position(), m.position() + m.length())) continue;
            std::string mm = m[1]; std::transform(mm.begin(), mm.end(), mm.begin(), ::tolower);
            int mo = month_index(mm.substr(0, 3));
            spans.push_back({(size_t)m.position(), (size_t)(m.position() + m.length()),
                             {iso(DEFAULT_YEAR, mo, 1), iso(DEFAULT_YEAR, mo, days_in_month(DEFAULT_YEAR, mo))}});
        }
    }
    // 4. Alphanumeric id (has a digit AND a letter)
    {
        std::regex re(R"(\b[A-Za-z_0-9]{2,}\b)");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
            auto m = *it;
            std::string s = m.str();
            bool hasd = false, hasa = false;
            for (char c : s) { if (isdigit((unsigned char)c)) hasd = true; if (isalpha((unsigned char)c)) hasa = true; }
            if (!(hasd && hasa)) continue;
            if (overlaps(m.position(), m.position() + m.length())) continue;
            spans.push_back({(size_t)m.position(), (size_t)(m.position() + m.length()), {s}});
        }
    }
    // 5. Standalone integers (not 0/1)
    {
        std::regex re(R"(\b\d+\b)");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
            auto m = *it;
            std::string s = m.str();
            if (s == "0" || s == "1") continue;
            if (overlaps(m.position(), m.position() + m.length())) continue;
            spans.push_back({(size_t)m.position(), (size_t)(m.position() + m.length()), {s}});
        }
    }

    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b){ return a.start < b.start; });

    std::string out; size_t last = 0;
    for (auto& sp : spans) {
        if (slot_values.size() + sp.vals.size() > 10) break;
        out += text.substr(last, sp.start - last);
        for (auto& v : sp.vals) {
            slot_values.push_back(v);
            out += " [val" + std::to_string(slot_values.size()) + "] ";
        }
        last = sp.end;
    }
    out += text.substr(last);
    return out;
}

static std::string relex(std::string sql, const std::vector<std::string>& slot_values) {
    for (int i = (int)slot_values.size(); i >= 1; --i) {
        std::string tok = "[val" + std::to_string(i) + "]";
        size_t pos = 0;
        while ((pos = sql.find(tok, pos)) != std::string::npos) {
            sql.replace(pos, tok.length(), slot_values[i - 1]);
            pos += slot_values[i - 1].length();
        }
    }
    sql = std::regex_replace(sql, std::regex(R"(\[val\d+\])"), "?");
    return sql;
}

// The C++ BPETokenizer.decode appends a space after every token, so a quoted
// value comes out as "' ABC123 '". Strip whitespace just INSIDE each quote
// pair so it matches in SQL, while preserving internal spaces of multi-word
// literals like 'SAFE EXPRESS' ([^']*? matches a single quoted run).
static std::string tidy_sql(const std::string& sql) {
    return std::regex_replace(sql, std::regex(R"('\s*([^']*?)\s*')"), "'$1'");
}

// ---- fusion.bin memory loader (header + global memory only) ---------------
static void load_memory(const std::string& path, int& seq_len, int& V, int& V_bpe,
                        int& M, int& mt, std::vector<float>& emit, std::vector<float>& toks, std::vector<float>& types)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Could not open " + path);
    int hdr[9]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    seq_len = hdr[2]; V = hdr[3]; V_bpe = hdr[4]; M = hdr[5]; mt = hdr[6];
    emit.resize(M);  f.read(reinterpret_cast<char*>(emit.data()), M * sizeof(float));
    toks.resize((size_t)M * mt); f.read(reinterpret_cast<char*>(toks.data()), toks.size() * sizeof(float));
    types.resize(M); f.read(reinterpret_cast<char*>(types.data()), M * sizeof(float));
    f.close();
}

static void load_expansions(const std::string& path, std::map<int, std::string>& exp) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        size_t p = line.find('|');
        if (p != std::string::npos) exp[std::stoi(line.substr(0, p))] = line.substr(p + 1);
    }
}

// Build input ids from a delexed (lowercased) question: encode text segments
// with BPE, splice slot ids manually (the C++ BPE tokenizer can't emit the
// bracketed [valN] specials atomically), append [sep]=3.
static std::vector<int> build_input(const std::string& delexed_lower, BPETokenizer& tok) {
    std::vector<int> ids;
    std::regex slot_re(R"(\[val(\d+)\])");
    auto begin = std::sregex_iterator(delexed_lower.begin(), delexed_lower.end(), slot_re);
    auto end = std::sregex_iterator();
    size_t last = 0;
    for (auto it = begin; it != end; ++it) {
        auto m = *it;
        std::string seg = delexed_lower.substr(last, m.position() - last);
        if (!seg.empty()) { auto e = tok.encode(seg, -1); ids.insert(ids.end(), e.begin(), e.end()); }
        int slot_n = std::stoi(m[1]);
        ids.push_back(5 + (slot_n - 1)); // [val1] = id 5
        last = m.position() + m.length();
    }
    std::string tail = delexed_lower.substr(last);
    if (!tail.empty()) { auto e = tok.encode(tail, -1); ids.insert(ids.end(), e.begin(), e.end()); }
    ids.push_back(3); // [sep]
    return ids;
}

int main(int argc, char** argv)
{
    try {
        int seq_len, V, V_bpe, M, mt;
        std::vector<float> emit, toks, types;
        load_memory("data/fusion.bin", seq_len, V, V_bpe, M, mt, emit, toks, types);

        std::map<int, std::string> expansions;
        load_expansions("data/fusion_expansions.txt", expansions);

        BPETokenizer tokenizer("data/bpe_vocab.txt", "data/bpe_merges.txt");

        // Frozen production config (must match the checkpoint). Override via
        // argv: infer_deep_fusion [dim] [depth].
        int dim = (argc > 1) ? std::stoi(argv[1]) : 512;
        int depth = (argc > 2) ? std::stoi(argv[2]) : 8;
        int heads = 8;

        DeepFusionNet model(V, V_bpe, seq_len, dim, heads, depth, 0.15f);
        model.set_memory(Tensor::upload(toks, {M, mt}), Tensor::upload(types, {1, M}), Tensor::upload(emit, {M, 1}));
        model.set_mode(false);
        model.load("weights/deep_fusion.bin");
        model.freeze_memory(); // encode the 859-row bank once, not per token

        std::cout << "\n=== DeepFusionNet Inference (C++) — V=" << V << " M=" << M << " ===\n";
        std::cout << "Type a question, or 'exit'.\n";

        const int eos_id = 4;
        while (true) {
            std::string q;
            std::cout << "\nUSER > ";
            if (!std::getline(std::cin, q) || q == "exit" || q.empty()) break;

            std::vector<std::string> slot_values;
            std::string delexed = extract_slots(q, slot_values);
            std::transform(delexed.begin(), delexed.end(), delexed.begin(), ::tolower);

            std::vector<int> seq = build_input(delexed, tokenizer);
            int start = (int)seq.size();

            std::vector<int> gen;
            for (int step = 0; step + start < seq_len; ++step) {
                int cur = (int)seq.size();
                std::vector<float> X(cur);
                for (int i = 0; i < cur; ++i) X[i] = (float)seq[i];
                Tensor dX = Tensor::upload(X, {1, cur});
                Tensor pred = model.forward(dX);      // [1, cur, V]
                std::vector<float> probs = pred.download();

                int base = (cur - 1) * V, best = 0; float bp = -1.0f;
                for (int v = 0; v < V; ++v) { float p = probs[base + v]; if (p > bp) { bp = p; best = v; } }
                if (best == eos_id || best == 0) break;
                gen.push_back(best);
                seq.push_back(best);
            }

            // detokenize: ids < V_bpe via BPE; ids >= V_bpe via expansion table
            std::string sql; std::vector<int> buf;
            auto flush = [&](){ if (!buf.empty()) { sql += tokenizer.decode(buf); buf.clear(); } };
            for (int id : gen) {
                if (id >= V_bpe) { flush(); auto e = expansions.find(id); if (e != expansions.end()) sql += e->second + " "; }
                else buf.push_back(id);
            }
            flush();

            std::cout << "SQL  > " << tidy_sql(relex(sql, slot_values)) << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
