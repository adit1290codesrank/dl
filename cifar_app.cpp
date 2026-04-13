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
#include "include/layers/augment.h"
#include "include/layers/softmax.h"
#include "include/layers/reshape.h"
#include "include/layers/dropout.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: cifar_app.exe <preprocessed_image.bin>" << std::endl;
        std::cerr << "  image.bin: 32*32*3 = 3072 raw float32 values in HWC order, normalized [0,1]" << std::endl;
        return 1;
    }
 
    std::ifstream f(argv[1], std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "ERROR: Cannot open input file: " << argv[1] << std::endl;
        return 1;
    }
 
    const int input_size = 32 * 32 * 3;
    std::vector<float> img(input_size);
    f.read(reinterpret_cast<char*>(img.data()), input_size * sizeof(float));
 
    if (f.gcount() != input_size * (int)sizeof(float))
    {
        std::cerr << "ERROR: Input file is wrong size. Expected " << input_size * sizeof(float) << " bytes." << std::endl;
        return 1;
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

    net.load("cifar10_weights.bin"); 

    Tensor d_input = Tensor::upload(img, 1, input_size);
    Tensor d_output = net.predict(d_input);
    std::vector<float> probs = d_output.download();
 
    const char* classes[10] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
 
    std::cout << "{";
    for (int i = 0; i < 10; i++)
    {
        std::cout << "\"" << classes[i] << "\":" << probs[i];
        if (i < 9) std::cout << ",";
    }
    std::cout << "}" << std::endl;
 
    return 0;
}