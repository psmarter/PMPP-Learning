# 第二十章 Exercise01 Makefile
# MPI + CUDA 分布式模板计算

# 编译器设置
NVCC = nvcc
# 使用系统 MPI 编译器和运行器（避免 Anaconda 冲突）
MPICXX = /usr/bin/mpicxx
MPIRUN = /usr/bin/mpirun

# GPU 架构 (根据你的 GPU 调整)
CUDA_ARCH = sm_89

# 编译标志
NVCC_FLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)
COMMON_DIR = ../../../Common

# MPI 包含路径和链接库
MPI_INCLUDE = $(shell mpicxx --showme:compile 2>/dev/null || echo "-I/usr/lib/x86_64-linux-gnu/openmpi/include")
MPI_LINK = $(shell mpicxx --showme:link 2>/dev/null || echo "-L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi")

# 目标
TARGET = stencil_mpi

# 源文件
SOURCES = solution.cu test.cpp

# 默认目标
all: $(TARGET)

# 编译规则：使用 nvcc 且指定 mpicxx 作为宿主编译器
$(TARGET): $(SOURCES) solution.h
	$(NVCC) $(NVCC_FLAGS) -I$(COMMON_DIR) -ccbin $(MPICXX) $(SOURCES) -o $(TARGET)

# 运行
run: $(TARGET)
	$(MPIRUN) -np 3 ./$(TARGET)

run2: $(TARGET)
	$(MPIRUN) -np 2 ./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o
	@echo "Cleaned all build files"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run the program"
	@echo "  make run2   - Build and run with 2 processes"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"

.PHONY: all run run2 clean help
