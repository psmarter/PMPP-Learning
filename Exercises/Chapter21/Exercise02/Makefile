# 第二十一章 Exercise02 Makefile
# 四叉树递归构建（动态并行）

# 编译器设置
NVCC = nvcc

# GPU 架构（根据你的 GPU 调整，动态并行需要 sm_35+）
# RTX 40 系列: sm_89
# RTX 30 系列: sm_86
# RTX 20 系列: sm_75
CUDA_ARCH = sm_89

# 动态并行编译选项
NVCC_FLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -rdc=true
LIBS = -lcudadevrt
COMMON_DIR = ../../../Common

# 目标
TARGET = quadtree_test

# 源文件
SOURCES = solution.cu test.cpp

# 默认目标
all: $(TARGET)

# 编译
$(TARGET): $(SOURCES) solution.h
	$(NVCC) $(NVCC_FLAGS) -I$(COMMON_DIR) $(SOURCES) $(LIBS) -o $(TARGET)

# 运行
run: $(TARGET)
	./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o
	@echo "Cleaned all build files"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run the program"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"

.PHONY: all run clean help
