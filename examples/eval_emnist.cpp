#include "include/core/tensor.h"
#include "include/core/network.h"
#include "include/layers/conv2d.h"
#include "include/layers/pooling.h"
#include "include/layers/dense.h"
#include "include/layers/activation.h"
#include "include/layers/batchnorm1d.h"
#include "include/layers/batchnorm2d.h" 
#include "include/layers/softmax.h"
#include "include/layers/reshape.h" 
#include "include/layers/dropout.h"
#include "include/core/loss.h"
#include "include/utils/data_loader.h" 
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

int main()
{
    std::vector<float> X_test, Y_test;
    int test_samples = 0;
    int classes = 47; 
    const int input_size = 28 * 28;

    std::cout << "Loading EMNIST dataset..." << std::endl;
    if (!load_emnist("./data/emnist-balanced-test-images-idx3-ubyte", "./data/emnist-balanced-test-labels-idx1-ubyte", X_test, Y_test, test_samples, classes)) 
    {
        std::cerr << "CRITICAL ERROR: Could not load EMNIST test dataset files!" << std::endl;
        return -1;
    }
    std::cout << "Successfully loaded " << test_samples << " images." << std::endl;

    Network ConvNet;

    ConvNet.add(std::make_unique<Reshape>(std::vector<int>{28,28,1})); 

    ConvNet.add(std::make_unique<Conv2D>(1,32,3,1,1)); 
    ConvNet.add(std::make_unique<BatchNorm2D>(32));
    ConvNet.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU,0.01f)); 
    ConvNet.add(std::make_unique<Pooling>(2,2));             
    
    ConvNet.add(std::make_unique<Conv2D>(32,64,3,1,1)); 
    ConvNet.add(std::make_unique<BatchNorm2D>(64));
    ConvNet.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU,0.01f)); 
    ConvNet.add(std::make_unique<Pooling>(2,2)); 

    ConvNet.add(std::make_unique<Conv2D>(64,128,3,1,1)); 
    ConvNet.add(std::make_unique<BatchNorm2D>(128));
    ConvNet.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU,0.01f)); 

    ConvNet.add(std::make_unique<Reshape>(std::vector<int>{6272}));

    ConvNet.add(std::make_unique<Dense>(6272,512)); 
    ConvNet.add(std::make_unique<BatchNorm1D>(512)); 
    ConvNet.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU,0.01f)); 
    ConvNet.add(std::make_unique<Dropout>(0.4f)); 

    ConvNet.add(std::make_unique<Dense>(512,47)); 
    ConvNet.add(std::make_unique<Softmax>());

    try {
        ConvNet.load("weights/emnist_t4_weights.bin");
    } catch(const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return -1;
    }

    auto evaluate = [&](const std::vector<float>& X, const std::vector<float>& Y, int samples, const std::string& dataset_name) {
        int batch_size = 256;
        int correct = 0;
        int num_batches = (samples + batch_size - 1) / batch_size;

        for (int b = 0; b < num_batches; b++) {
            int current_batch_size = std::min(batch_size, samples - b * batch_size);
            std::vector<float> batchX(current_batch_size * input_size);
            
            std::copy(X.begin() + b * batch_size * input_size, 
                      X.begin() + (b * batch_size + current_batch_size) * input_size, 
                      batchX.begin());

            Tensor d_X = Tensor::upload(batchX, current_batch_size, input_size);
            Tensor out = ConvNet.predict(d_X); 
            std::vector<float> predictions = out.download();

            for (int i = 0; i < current_batch_size; i++) {
                int pred_class = 0;
                float max_val = predictions[i * classes];
                for (int j = 1; j < classes; j++) {
                    if (predictions[i * classes + j] > max_val) {
                        max_val = predictions[i * classes + j];
                        pred_class = j;
                    }
                }

                int true_class = 0;
                float max_true = Y[(b * batch_size + i) * classes];
                for (int j = 1; j < classes; j++) {
                    if (Y[(b * batch_size + i) * classes + j] > max_true) {
                        max_true = Y[(b * batch_size + i) * classes + j];
                        true_class = j;
                    }
                }

                if (pred_class == true_class) correct++;
            }
            std::cout << "\rEvaluating " << dataset_name << " [" << std::setw(3) << b + 1 << "/" << num_batches << "]..." << std::flush;
        }
        std::cout << "\n" << dataset_name << " Accuracy: " << (float)correct / (float)samples * 100.0f << "%\n\n";
    };

    std::cout << "\nStarting Evaluation...\n";
    evaluate(X_test, Y_test, test_samples, "Testing Dataset");

    return 0;
}
