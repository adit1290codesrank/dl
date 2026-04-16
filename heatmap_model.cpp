#include "include/core/network.h"
#include "include/layers/embedding.h"
#include "include/layers/transformer.h"
#include "include/layers/reshape.h"
#include "include/layers/dense.h"
#include "include/layers/activation.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <numeric>   // Added for std::iota
#include <algorithm> // Added for std::shuffle
#include <random>    // Added for std::mt19937

std::unique_ptr<Network> create_predictive_heatmap_network(int num_features, int total_vocab_size, int embed_dim, int heads, int num_transformer_layers) {
    auto net = std::make_unique<Network>();
    net->add(std::make_unique<Embedding>(total_vocab_size, embed_dim));

    for (int i = 0; i < num_transformer_layers; ++i) {
        net->add(std::make_unique<Transformer>(embed_dim, heads));
    }

    net->add(std::make_unique<Reshape>(std::vector<int>{num_features * embed_dim}));
    
    // Non-Linear Head
    net->add(std::make_unique<Dense>(num_features * embed_dim, 64));
    net->add(std::make_unique<Activation>(ActivationType::LEAKY_RELU, 0.01f));
    net->add(std::make_unique<Dense>(64, 1));
    net->add(std::make_unique<Activation>(ActivationType::SIGMOID));

    return net;
}

int main(int argc, char* argv[]) {
    // -------------------------------------------------------------
    // HYPERPARAMETERS
    // -------------------------------------------------------------
    int num_features = 4; 
    int total_vocab_size = 41; // <-- LOCKED IN FROM YOUR PYTHON SCRIPT
    int embed_dim = 32;
    int heads = 4;
    int num_transformer_layers = 2;
    
    int epochs = 150; // Increased to 500 to let it fully converge
    int batch_size = 64;
    float learning_rate = 0.0001f; // 5e-5: Slightly faster but still safe

    std::cout << "-------------------------------------------\n";
    std::cout << "Building Predictive Heatmap Tabular Network\n";
    std::cout << "Features: " << num_features << " | Total Vocab Size: " << total_vocab_size << "\n";
    std::cout << "-------------------------------------------\n";

    auto net = create_predictive_heatmap_network(num_features, total_vocab_size, embed_dim, heads, num_transformer_layers);
    std::cout << "Network built successfully!\n\n";

    // -------------------------------------------------------------
    // DATA INGESTION (Reading the real PostgreSQL dump)
    // -------------------------------------------------------------
    std::cout << "Loading real database rows from transformer_data.csv..." << std::endl;
    
    std::vector<float> X;
    std::vector<float> Y;
    
    std::ifstream file("transformer_data.csv");
    if(!file.is_open()) {
        std::cerr << "❌ ERROR: Failed to open transformer_data.csv!" << std::endl;
        std::cerr << "Did you copy it from the 'dbms' folder to the 'deep_learning' folder?" << std::endl;
        return 1;
    }

    std::string line, val;
    std::getline(file, line); // Skip the CSV header row

    int num_samples = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        
        // Read the 4 feature IDs (Hour, Day, Loc, Node)
        for(int i = 0; i < num_features; i++) {
            std::getline(ss, val, ',');
            X.push_back(std::stof(val));
        }
        // Read the target utilization rate
        std::getline(ss, val, ',');
        Y.push_back(std::stof(val));
        
        num_samples++;
    }
    file.close();

    std::cout << "✅ Successfully loaded " << num_samples << " rows from production database.\n\n";

    // -------------------------------------------------------------
    // CRITICAL FIX: SHUFFLE THE DATASET
    // -------------------------------------------------------------
    std::cout << "Shuffling dataset to remove chronological bias...\n";
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(42); // Fixed seed for reproducibility
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<float> X_shuffled(num_samples * num_features);
    std::vector<float> Y_shuffled(num_samples);

    for(int i = 0; i < num_samples; ++i) {
        int old_idx = indices[i];
        for(int j = 0; j < num_features; ++j) {
            X_shuffled[i * num_features + j] = X[old_idx * num_features + j];
        }
        Y_shuffled[i] = Y[old_idx];
    }

    X = X_shuffled;
    Y = Y_shuffled;

    // -------------------------------------------------------------
    // TRAINING PIPELINE
    // -------------------------------------------------------------
    int train_samples = (int)(num_samples * 0.8);
    int test_samples = num_samples - train_samples;

    std::vector<float> X_train(X.begin(), X.begin() + train_samples * num_features);
    std::vector<float> Y_train(Y.begin(), Y.begin() + train_samples);

    std::vector<float> X_test(X.begin() + train_samples * num_features, X.end());
    std::vector<float> Y_test(Y.begin() + train_samples, Y.end());

    std::cout << "Dataset Ready. Initializing Training Pipeline...\n" << std::endl;
    try {
        net->fit(X_train, Y_train, train_samples, num_features, 1, epochs, 10, learning_rate, batch_size, LossType::MSE);
        std::cout << "\nTraining phase finalized.\n" << std::endl;

        // -------------------------------------------------------------
        // EVALUATION PHASE
        // -------------------------------------------------------------
        std::cout << "-------------------------------------------\n";
        std::cout << "Final Model Evaluation (Test Set: " << test_samples << " rows)\n";
        std::cout << "-------------------------------------------\n";
        
        net->eval();
        Tensor d_test_X = Tensor::upload(X_test, test_samples, num_features);
        Tensor predictions_tensor = net->predict(d_test_X);
        std::vector<float> preds = predictions_tensor.download();

        float total_mae = 0.0f;
        float total_mse = 0.0f;
        float total_y_sum = 0.0f;
        int accurate_count = 0;
        float threshold = 0.15f; // 15% Error Threshold

        std::cout << "Sample Predictions (First 15 Rows):\n";
        std::cout << "ID | Predicted | Actual  | Abs Error | Status\n";
        std::cout << "---|-----------|---------|-----------|--------\n";
        
        for(int i = 0; i < test_samples; i++) {
            float p = preds[i];
            float a = Y_test[i];
            float err = std::abs(p - a);
            
            // Utilization rate is usually between 0.0 and 1.0. 
            // We adjust the relative error calculation slightly to avoid div-by-zero on completely empty hours.
            float rel_err = err / (std::abs(a) + 0.1f); 
            
            total_mae += err;
            total_mse += err * err;
            total_y_sum += a;
            
            if(rel_err <= threshold) accurate_count++;

            if(i < 15) {
                printf("%2d | %9.4f | %7.4f | %9.4f | %s\n", 
                       i, p, a, err, (rel_err <= threshold ? "PASS" : "FAIL"));
            }
        }

        float mae = total_mae / test_samples;
        float mse = total_mse / test_samples;
        float rmse = std::sqrt(mse);
        
        float mean_y = total_y_sum / test_samples;
        float ss_tot = 0.0f;
        for(int i = 0; i < test_samples; i++) {
            ss_tot += (Y_test[i] - mean_y) * (Y_test[i] - mean_y);
        }
        float r2 = (ss_tot == 0.0f) ? 0.0f : 1.0f - (total_mse / ss_tot);
        float accuracy = (float)accurate_count / test_samples * 100.0f;

        std::cout << "-------------------------------------------\n";
        std::cout << "DETAILED PERFORMANCE REPORT\n";
        std::cout << "-------------------------------------------\n";
        printf("Mean Absolute Error (MAE):  %.4f\n", mae);
        printf("Root Mean Sq. Error (RMSE): %.4f\n", rmse);
        printf("R-Squared (R2) Score:     %.4f\n", r2);
        printf("Accuracy (Within 15%%):    %.2f%%\n", accuracy);
        std::cout << "-------------------------------------------\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
    
    net->save("advanced_heatmap_weights.bin");
    std::cout << "✅ Model weights saved to advanced_heatmap_weights.bin\n";
    
    return 0;
}