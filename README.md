# cuDNN-Lite: High-Performance Custom C++/CUDA Deep Learning Engine

<div align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg" alt="C++17">
  <img src="https://img.shields.io/badge/CUDA-11.x%2B-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/Dependencies-Zero%20(Scratch)-red.svg" alt="No PyTorch">
</div>

cuDNN-Lite is a custom, high-performance Deep Learning framework built entirely from scratch in C++ and CUDA. It implements an autograd engine, tensor operations, and highly optimized custom CUDA kernels without relying on any external high-level frameworks like PyTorch or TensorFlow.

This repository contains the core framework as well as three state-of-the-art models trained natively on the engine for Computer Vision and Natural Language Processing.

## 🚀 Key Features

*   **Custom Autograd Engine:** Forward and backward pass tracking with dynamic computation graphs.
*   **Custom CUDA Kernels:** Hand-written, optimized kernels for Softmax, LayerNorm, Adam Optimizer, Max Pooling, and Cross-Entropy Loss.
*   **cuBLAS Integration:** High-performance batched matrix multiplications for attention mechanisms and dense layers.
*   **Zero High-Level Dependencies:** Built from the ground up. Python is only used for initial dataset downloading and tokenization preprocessing.

## 📊 Models & Benchmarks

All models were trained natively on the cuDNN-Lite engine utilizing an NVIDIA L4 Tensor Core GPU (Ada Lovelace `sm_89` architecture).

### 1. Vision: CIFAR-10 ResNet
A custom Convolutional Neural Network with Residual Blocks (ResNet-style) trained on 60,000 32x32 color images.
*   **Architecture:** Conv2D, LeakyReLU, ResBlocks, GlobalAveragePooling, Dense.
*   **Test Accuracy:** `93.53%`

### 2. Vision: EMNIST Balanced
A Deep CNN trained on over 100,000 handwritten characters and digits (47 classes).
*   **Architecture:** Deep Conv2D stack, BatchNorm2D, MaxPooling, Dropout.
*   **Test Accuracy:** `88.26%`

### 3. NLP: BERT-Base (CLINC150 Intent Classification)
A massive 85-million parameter Transformer model built entirely from scratch. Features standard Self-Attention mechanisms, LayerNorm, and custom Adam optimization to prevent vanishing gradients.
*   **Dataset:** CLINC150 (DeepPavlov) - 150 unique intent classes.
*   **Architecture:** 12 Heads, 12 Layers, 768 Hidden Dimension.
*   **Validation Accuracy:** `83.83%`

## 📁 Repository Structure

```text
cuDNN-Lite/
├── src/                # Core framework C++ and CUDA implementations
│   ├── core/           # Tensor math, cuBLAS wrappers, autograd logic
│   ├── layers/         # Neural network layers (Conv2D, Dense, Attention, etc.)
│   └── models/         # High-level architecture builders (BERT, ResNet)
├── include/            # Framework header files
├── examples/           # Training and evaluation entry points
│   ├── train_alexa.cpp # NLP training loop
│   ├── eval_alexa.cpp  # NLP evaluation
│   ├── train_cifar.cpp # Vision training loop
│   ├── eval_cifar.cpp  # Vision evaluation
│   ├── train_emnist.cpp# Character recognition training loop
│   ├── eval_emnist.cpp # Character recognition evaluation
│   └── cli.cpp         # Interactive C++ NLP Inference CLI
├── scripts/            # Python data preprocessing
└── Makefile            # sm_89 optimized build system
```

## 🛠️ Building and Running

Compile the framework and entry points using the provided Makefile. The build system is heavily optimized for NVIDIA Ada Lovelace architecture (`sm_89`).

```bash
# Example: Build and run CIFAR-10 evaluation
make clean && make MAIN_SRC=examples/eval_cifar.cpp ARCH=sm_89
./eval

# Example: Run the Interactive Universal AI CLI
make clean && make MAIN_SRC=examples/cli.cpp ARCH=sm_89
./cli
```

## 🧠 Interactive CLI

cuDNN-Lite includes a fully native C++ inference shell. After training the BERT model, you can run `./cli` to chat with the model and perform real-time intent classification with ultra-low latency directly on the GPU.
