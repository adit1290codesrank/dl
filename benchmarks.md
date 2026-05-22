# 🏆 Model Benchmarks & Achievements

This document tracks the performance, architectures, and hyperparameters of the deep learning models trained from scratch in this C++ CUDA framework.

---

## 🚀 CIFAR-10 Custom ResNet (CNN)
**Date**: May 2026  
**Final Test Accuracy**: **93.53%** 🎉 (Training Accuracy: 99.858%)  
**Training Time**: ~80 epochs  

### 🧠 Architecture Summary
A custom ResNet-inspired architecture built entirely from scratch in C++/CUDA.
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

### 📊 Training Highlights
* **Initial Loss**: ~2.60 (Epoch 1)
* **Final Training Loss**: ~1.03 (Epoch 80)
* **Generalization**: Excellent. The combination of strong data augmentation (Cutout/Cropping) and heavy regularization (Dropout + BatchNorm) allowed the model to achieve an incredible 93.53% accuracy on the test set.

---

## 🔮 Future Benchmarks (To-Do)
* [ ] **CIFAR-10 Vision Transformer (ViT)**
* [ ] **Tiny ImageNet (ViT / ResNet)**
* [ ] **Jena Climate Time-Series Forecasting (Transformer)**
* [ ] **EMNIST Handwriting Recognition**
