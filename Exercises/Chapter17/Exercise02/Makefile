# ============================================================================
# Makefile - 第十七章练习2: F^H D 核心计算
# ============================================================================

NVCC = nvcc

CUDA_ARCH = sm_89
CUDA_FLAGS = -arch=$(CUDA_ARCH) -O3
CXX_FLAGS = -std=c++17

COMMON_DIR = ../../../Common

TARGET = test_fhd

CUDA_SRC = solution.cu
CPP_SRC = test.cpp

OBJECTS = solution.o test.o

all: $(TARGET)
	@echo "Build completed successfully!"
	@echo "Run 'make run' to execute the program"

$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

solution.o: $(CUDA_SRC) solution.h
	$(NVCC) $(CUDA_FLAGS) -I$(COMMON_DIR) -c $<

test.o: $(CPP_SRC) solution.h
	$(NVCC) $(CXX_FLAGS) -I$(COMMON_DIR) -c $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS)
	@echo "Cleaned all build files"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run the program"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"

.PHONY: all run clean help
