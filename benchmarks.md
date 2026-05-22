# 🏆 Model Benchmarks & Achievements

This document tracks the performance, architectures, and hyperparameters of the deep learning models trained from scratch in this C++ CUDA framework.

---

## 🚀 CIFAR-10 Custom ResNet (CNN)
**Date**: May 2026  
**Final Test Accuracy**: **84.14%** 🎉  
**Training Time**: ~80 epochs  

### 🧠 Architecture Summary
* **Input**: 32x32x3 (RGB Images)
* **Stem**: `Conv2D(3 -> 64)` + `BatchNorm` + `LeakyReLU`
* **Stage 1**: 2x `ResBlock(64)` -> `MaxPooling(2x2)`
* **Stage 2**: `ProjResBlock(64 -> 128)` + 1x `ResBlock(128)` -> `MaxPooling(2x2)`
* **Stage 3**: `ProjResBlock(128 -> 256)` + 1x `ResBlock(256)`
* **Head**: `GlobalAvgPool()` -> `Dense(256 -> 128)` + `BatchNorm` + `LeakyReLU` + `Dropout(0.3)` -> `Dense(128 -> 10)` -> `Softmax`
* **Estimated Parameters**: **~2.8 Million**

### ⚙️ Hyperparameters
* **Optimizer**: Adam (CUDA accelerated)
* **Loss Function**: Categorical Cross-Entropy
* **Batch Size**: 128
* **Epochs**: 80
* **Learning Rate**: Cosine Annealing (Max: `0.001` -> Min: `1e-5`)
* **Data Augmentation**: 
  * Random Cropping (padding = 4)
  * Cutout (hole size = 8x8)
  * Random Horizontal Flips