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

int main()
{
    std::vector<float> X_train, Y_train;
    int samples=0;
    const int classes=10;
    const int input=32*32*3;

    std::vector<std::string> train_files = 
    {
        "./data/cifar-10-batches-bin/data_batch_1.bin",
        "./data/cifar-10-batches-bin/data_batch_2.bin",
        "./data/cifar-10-batches-bin/data_batch_3.bin",
        "./data/cifar-10-batches-bin/data_batch_4.bin",
        "./data/cifar-10-batches-bin/data_batch_5.bin",
    };

    if (!load_cifar10(train_files, X_train, Y_train, samples, classes))
    {
        std::cerr << "Failed to load CIFAR-10!" << std::endl;
        return -1;
    }

    Network net;

    net.add(std::make_unique<Reshape>(std::vector<int>{32, 32, 3}));
    net.add(std::make_unique<Augment>(32, 32, 3, 4, 8));

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

    net.fit(X_train, Y_train, samples, input, classes,
            80,       
            1,        
            0.001f,   
            128,      
            LossType::CROSS_ENTROPY);

    net.save("cifar10_weights.bin");
    return 0;
}