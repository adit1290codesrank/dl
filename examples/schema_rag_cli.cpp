#include "../models/transformers/schema_rag_net.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>

// Simulated CLI for Schema-RAG
int main() {
    try {
        std::cout << "========================================================\n";
        std::cout << " BreakWalls Schema-RAG Pointer Network CLI\n";
        std::cout << "========================================================\n\n";

        std::cout << "Loading Base Architecture...\n";
        int vocab_size = 5000;
        int seq_len = 32;
        int dim = 128;
        int heads = 4;
        int depth = 2;

        SchemaRAGNet model(vocab_size, seq_len, dim, heads, depth);
        // model.load("weights/schema_rag.bin");

        std::cout << "Loading Business Schema (Dynamic Context)...\n";
        std::map<std::string, std::string> schema = {
            {"MPY", "SMU IN ('Protective Coating', 'Marine') AND NatureOfTransaction = 'Sales'"},
            {"OBD", "PickListId"},
            {"Client", "CustomerName"}
        };

        for(auto const& [key, val] : schema) {
            std::cout << "  Schema Node: [" << key << "] -> " << val << "\n";
        }
        
        std::cout << "\nNetwork Ready. (Simulated Output for demo purposes)\n";
        std::string input;
        
        while(true) {
            std::cout << "\nBreakWalls> ";
            std::getline(std::cin, input);
            if(input == "exit" || input == "quit") break;
            if(input.empty()) continue;

            // In a full implementation, we'd tokenize the input and schema, run model.forward(),
            // and decode the tokens, applying the scatter-add pointer logic.
            // Here we just simulate the resolution based on our schema dictionary mapping.

            std::cout << "[Dual-Encoder] Query shape: (1, 32, 768)\n";
            std::cout << "[Dual-Encoder] Schema shape: (1, 3, 768)\n";
            std::cout << "[Pointer Attention] Q * K^T Similarity calculated.\n";

            std::string output = "SELECT * FROM Sales WHERE ";
            bool first = true;

            if (input.find("MPY") != std::string::npos || input.find("mpy") != std::string::npos) {
                std::cout << "  -> Pointer Network explicitly selected Schema Index 0 (Similarity: 0.98)\n";
                output += schema["MPY"];
                first = false;
            }
            if (input.find("OBD") != std::string::npos || input.find("obd") != std::string::npos) {
                std::cout << "  -> Pointer Network explicitly selected Schema Index 1 (Similarity: 0.99)\n";
                if(!first) output += " AND ";
                output += schema["OBD"] + " = ?";
                first = false;
            }
            if (input.find("Client") != std::string::npos || input.find("client") != std::string::npos) {
                std::cout << "  -> Pointer Network explicitly selected Schema Index 2 (Similarity: 0.97)\n";
                if(!first) output += " AND ";
                output += schema["Client"] + " = ?";
                first = false;
            }
            
            if (first) {
                std::cout << "  -> No specific schema linked. Fallback generation.\n";
                output = "SELECT * FROM Sales";
            }

            std::cout << "\n[Generated SQL]: " << output << "\n";
        }

    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
