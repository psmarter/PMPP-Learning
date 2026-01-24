# 第十八章 Exercise01 Makefile
# 静电势能图计算

NVCC = nvcc

# CUDA 架构（根据你的 GPU 调整）
# RTX 40 系列: sm_89
# RTX 30 系列: sm_86
# RTX 20 系列: sm_75
CUDA_ARCH = sm_89
CUDA_FLAGS = -arch=$(CUDA_ARCH) -O3
CXX_FLAGS = -std=c++17
COMMON_DIR = ../../../Common

# 目标可执行文件
TARGET = cenergy_benchmark

# 源文件
SOURCES = solution.cu test.cpp

# 默认目标
all: $(TARGET)

# 编译
$(TARGET): $(SOURCES) solution.h
	$(NVCC) $(CUDA_FLAGS) -I$(COMMON_DIR) $(SOURCES) -o $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o
	@echo "Cleaned all build files"

# Windows 清理
clean-win:
	del /F /Q $(TARGET).exe 2>nul || echo.

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run the program"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"

.PHONY: all run clean clean-win help
