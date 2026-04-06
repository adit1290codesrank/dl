# ==========================================
# Compilers and Flags
# ==========================================
CXX      := g++
NVCC     := nvcc

# Path to CUDA installation
CUDA_PATH := /usr/local/cuda

# We add the 'targets' path because g++ needs to know exactly where the 
# x86_64 linux headers and libraries are hiding.
CUDA_INC  := -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/x86_64-linux/include
CUDA_LIB  := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib

CXXFLAGS  := -O3 -std=c++17 -Iinclude $(CUDA_INC)
NVCCFLAGS := -O3 -std=c++17 -Iinclude -arch=sm_75
LDFLAGS   := $(CUDA_LIB) -lcublas -lcudart

# ==========================================
# Project Structure
# ==========================================
BUILD_DIR := build
SRC_DIR   := src
OBJ_DIR   := $(BUILD_DIR)/obj

# Source files (Dynamically finds all .cpp and .cu files)
CPP_SRCS := $(wildcard $(SRC_DIR)/core/*.cpp) \
            $(wildcard $(SRC_DIR)/layers/*.cpp) \
            main.cpp

CU_SRCS  := $(wildcard $(SRC_DIR)/core/*.cu)

# Object files (Maps .cpp and .cu files to .o files in the build dir)
OBJS     := $(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o) \
            $(CU_SRCS:%.cu=$(OBJ_DIR)/%.o)

# Final Binary
TARGET   := main

# ==========================================
# Rules
# ==========================================

all: $(TARGET)

# Link everything together 
# (Notice we removed main.cpp here, we only link the .o files!)
$(TARGET): $(OBJS)
	@echo "Linking binary: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile C++ source files
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling C++: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Cleanup
clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR) $(TARGET)

# Helper to run the test
run: all
	./$(TARGET)

.PHONY: all clean run