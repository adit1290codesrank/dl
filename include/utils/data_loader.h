#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>


uint32_t swap_endian(uint32_t val) 
{
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

bool load_emnist(const std::string& image_path, const std::string& label_path, std::vector<float>& X, std::vector<float>& Y, int& num_samples, int num_classes = 47) {
    
    std::ifstream file_img(image_path, std::ios::binary);
    std::ifstream file_lbl(label_path, std::ios::binary);

    if (!file_img.is_open() || !file_lbl.is_open()) 
    {
        std::cerr << "Failed to open EMNIST files." << std::endl;
        return false;
    }

    uint32_t magic_img, magic_lbl, num_img, num_lbl, rows, cols;

    file_img.read(reinterpret_cast<char*>(&magic_img), 4);
    file_img.read(reinterpret_cast<char*>(&num_img), 4);
    file_img.read(reinterpret_cast<char*>(&rows), 4);
    file_img.read(reinterpret_cast<char*>(&cols), 4);

    file_lbl.read(reinterpret_cast<char*>(&magic_lbl), 4);
    file_lbl.read(reinterpret_cast<char*>(&num_lbl), 4);

    num_img = swap_endian(num_img);
    num_lbl = swap_endian(num_lbl);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (num_img != num_lbl) 
    {
        std::cerr << "Mismatch between image and label counts." << std::endl;
        return false;
    }

    num_samples = num_img;
    int img_size = rows * cols; 

    X.resize(num_samples * img_size);
    Y.assign(num_samples * num_classes, 0.0f); 

    std::vector<uint8_t> raw_img(img_size);
    uint8_t raw_lbl;

    for (int i = 0; i < num_samples; i++) 
    {
        file_img.read(reinterpret_cast<char*>(raw_img.data()), img_size);
        file_lbl.read(reinterpret_cast<char*>(&raw_lbl), 1);

        if (raw_lbl < num_classes) 
        {
            Y[i * num_classes + raw_lbl] = 1.0f;
        }

        for (int r = 0; r < rows; r++) 
        {
            for (int c = 0; c < cols; c++) 
            {
                float pixel = raw_img[c * rows + r] / 255.0f;
                X[i * img_size + (r * cols + c)] = pixel;
            }
        }
    }
    return true;
}