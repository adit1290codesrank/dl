CXX      := g++
NVCC     := nvcc

CUDA_PATH := /usr/local/cuda
CUDA_INC  := -I$(CUDA_PATH)/include -I$(CUDA_PATH)/targets/x86_64-linux/include
CUDA_LIB  := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/targets/x86_64-linux/lib

ARCH     ?= sm_75
MAIN_SRC ?= examples/test.cpp
TARGET   := $(notdir $(basename $(MAIN_SRC)))

CXXFLAGS  := -O3 -std=c++17 -I. -Iinclude $(CUDA_INC)
NVCCFLAGS := -O3 -std=c++17 -I. -Iinclude -arch=$(ARCH)
LDFLAGS   := $(CUDA_LIB) -lcublas -lcudart

BUILD_DIR := build
SRC_DIR   := src
OBJ_DIR   := $(BUILD_DIR)/obj

CPP_SRCS := $(wildcard $(SRC_DIR)/core/*.cpp) \
            $(wildcard $(SRC_DIR)/layers/*.cpp) \
            $(MAIN_SRC)

CU_SRCS  := $(wildcard $(SRC_DIR)/core/*.cu) \
            $(wildcard $(SRC_DIR)/layers/*.cu)

OBJS     := $(CPP_SRCS:%.cpp=$(OBJ_DIR)/%.o) \
            $(CU_SRCS:%.cu=$(OBJ_DIR)/%.o)

all: dirs $(TARGET)

dirs:
	@mkdir -p weights outputs

$(TARGET): $(OBJS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "CXX: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	@echo "NVCC: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run dirs