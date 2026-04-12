#include "include/core/network.h"
#include "include/layers/conv2d.h"
#include "include/layers/pooling.h"
#include "include/layers/dense.h"
#include "include/layers/resblock.h"
#include "include/layers/projresblock.h"
#include "include/layers/gap.h"
#include "include/layers/activation.h"
#include "include/layers/batchnorm1d.h"
#include "include/layers/batchnorm2d.h"
#include "include/layers/softmax.h"
#include "include/layers/reshape.h"
#include "include/layers/dropout.h"
#include "include/core/loss.h"
#include "include/utils/cifar_loader.h"
#include "include/layers/augment.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

int main()
{
    std::vector<float> X_train, Y_train;
    std::vector<float> X_test, Y_test;
    int train_samples = 0;
    int test_samples = 0;
    const int classes = 10;
    const int input_size = 32 * 32 * 3;

    std::vector<std::string> train_files = 
    {
        "./data/cifar-10-batches-bin/data_batch_1.bin",
        "./data/cifar-10-batches-bin/data_batch_2.bin",
        "./data/cifar-10-batches-bin/data_batch_3.bin",
        "./data/cifar-10-batches-bin/data_batch_4.bin",
        "./data/cifar-10-batches-bin/data_batch_5.bin",
    };

    std::vector<std::string> test_files = 
    {
        "./data/cifar-10-batches-bin/test_batch.bin"
    };

    if (!load_cifar10(train_files, X_train, Y_train, train_samples, classes))
    {
        std::cerr << "Failed to load CIFAR-10 training data!" << std::endl;
        return -1;
    }

    if (!load_cifar10(test_files, X_test, Y_test, test_samples, classes))
    {
        std::cerr << "Failed to load CIFAR-10 testing data!" << std::endl;
        return -1;
    }

    Network net;

    // Same architecture as cifar.cpp
    net.add(std::make_unique<Reshape>(std::vector<int>{32, 32, 3}));
    net.add(std::make_unique<Augment>(32, 32, 3, 4, 8)); // Keeping it to maintain layer alignment during load()

    net.add(std::make_unique<Conv2D>(3, 64, 3, 1, 1));
    net.add(std::make_unique<BatchNorm2D>(64));
    net.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU, 0.01f));

    net.add(std::make_unique<ResBlock>(64));
    net.add(std::make_unique<ResBlock>(64));
    net.add(std::make_unique<Pooling>(2, 2));  

    net.add(std::make_unique<ProjResBlock>(64, 128));
    net.add(std::make_unique<ResBlock>(128));
    net.add(std::make_unique<Pooling>(2, 2));  

    net.add(std::make_unique<ProjResBlock>(128, 256));
    net.add(std::make_unique<ResBlock>(256));

    net.add(std::make_unique<GlobalAvgPool>());          
    net.add(std::make_unique<Reshape>(std::vector<int>{256}));
    net.add(std::make_unique<Dense>(256, 128));
    net.add(std::make_unique<BatchNorm1D>(128));
    net.add(std::make_unique<Activation>(ActivationType::LEAKY_RELU, 0.01f));
    net.add(std::make_unique<Dropout>(0.3f));
    net.add(std::make_unique<Dense>(128, 10));
    net.add(std::make_unique<Softmax>());

    try {
        net.load("cifar10_weights.bin");
    } catch(const std::exception& e) {
        std::cerr << "Error loading weights: " << e.what() << std::endl;
        return -1;
    }

    auto evaluate = [&](const std::vector<float>& X, const std::vector<float>& Y, int samples, const std::string& dataset_name) {
        int batch_size = 256; // Adjust if necessary for optimal VRAM usage (6GB should comfortably handle 256)
        int correct = 0;
        int num_batches = (samples + batch_size - 1) / batch_size;

        for (int b = 0; b < num_batches; b++) {
            int current_batch_size = std::min(batch_size, samples - b * batch_size);
            std::vector<float> batchX(current_batch_size * input_size);
            
            std::copy(X.begin() + b * batch_size * input_size, 
                      X.begin() + (b * batch_size + current_batch_size) * input_size, 
                      batchX.begin());

            Tensor d_X = Tensor::upload(batchX, current_batch_size, input_size);
            Tensor out = net.predict(d_X); 
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
    evaluate(X_train, Y_train, train_samples, "Training Dataset");
    evaluate(X_test, Y_test, test_samples, "Testing Dataset");

    return 0;
}
