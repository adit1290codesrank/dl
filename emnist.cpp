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

int main()
{
    std::vector<float> X_train, Y_train;
    int samples=0;
    int classes=47; 
    int input=28*28;


    std::cout << "Loading dataset..." << std::endl;
    if (!load_emnist("./data/emnist-balanced-train-images-idx3-ubyte", "./data/emnist-balanced-train-labels-idx1-ubyte", X_train, Y_train, samples, classes)) 
    {
        std::cerr << "CRITICAL ERROR: Could not load EMNIST dataset files!" << std::endl;
        return -1;
    }
    std::cout << "Successfully loaded " << samples << " images." << std::endl;

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

    ConvNet.fit(
        X_train, 
        Y_train, 
        samples, 
        input, 
        classes, 
        30,
        1,      
        0.001f,
        64,           
        LossType::CROSS_ENTROPY
    ); 

    std::cout << "\nTraining Complete! Saving weights..." << std::endl;
    ConvNet.save("emnist_t4_weights.bin");
    std::cout << "Weights successfully saved to emnist_weights.bin" << std::endl;

    return 0;
}