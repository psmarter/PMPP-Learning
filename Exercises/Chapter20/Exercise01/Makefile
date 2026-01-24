# 第二十章 Exercise01 Makefile
# MPI + CUDA 分布式模板计算

# 编译器设置
NVCC = nvcc
MPICXX = mpicxx

# GPU 架构 (根据你的 GPU 调整)
CUDA_ARCH = sm_60

# 编译标志
NVCC_FLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)

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
	$(NVCC) $(NVCC_FLAGS) -ccbin $(MPICXX) $(SOURCES) -o $(TARGET)

# 运行
run: $(TARGET)
	mpirun -np 3 ./$(TARGET)

run2: $(TARGET)
	mpirun -np 2 ./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o

.PHONY: all run run2 clean
