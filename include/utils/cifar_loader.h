#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

bool load_cifar10(const std::vector<std::string>& batch_paths,std::vector<float>& X,std::vector<float>& Y,int& num_samples,int num_classes=10)
{
    X.clear();
    Y.clear();
    num_samples = 0;

    for (const auto& path : batch_paths)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
        {
            std::cerr << "Failed to open: " << path << std::endl;
            return false;
        }

        while (f.peek() != EOF)
        {
            uint8_t label;
            if (!f.read(reinterpret_cast<char*>(&label), 1)) break;

            const int img_bytes = 3072;
            std::vector<uint8_t> raw(img_bytes);
            f.read(reinterpret_cast<char*>(raw.data()), img_bytes);

            std::vector<float> one_hot(num_classes, 0.0f);
            one_hot[label] = 1.0f;
            Y.insert(Y.end(), one_hot.begin(), one_hot.end());

            std::vector<float> hwc(img_bytes);
            for (int c = 0; c < 3; c++)
                for (int h = 0; h < 32; h++)
                    for (int w = 0; w < 32; w++)
                        hwc[(h * 32 + w) * 3 + c] = raw[c * 1024 + h * 32 + w] / 255.0f;

            X.insert(X.end(), hwc.begin(), hwc.end());
            num_samples++;
        }
    }

    std::cout << "Loaded " << num_samples << " CIFAR-10 images." << std::endl;
    return true;
}