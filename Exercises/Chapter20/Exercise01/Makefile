# 第二十章 Exercise01 Makefile
# MPI + CUDA 分布式模板计算

# 使用 mpicc 包装 nvcc
# OMPI_CXX 告诉 OpenMPI 使用 nvcc 作为底层 C++ 编译器
NVCC = nvcc
MPICXX = mpicxx
CUDA_FLAGS = -O3 -std=c++17

# 检测 GPU 架构 (默认 sm_60)
CUDA_ARCH ?= sm_60

# 目标可执行文件
TARGET = stencil_mpi

# 源文件
CU_SOURCES = solution.cu
CPP_SOURCES = test.cpp

# 编译选项
NVCC_FLAGS = $(CUDA_FLAGS) -arch=$(CUDA_ARCH)

# 默认目标
all: $(TARGET)

# 方法1：直接使用 nvcc 编译所有文件（推荐）
$(TARGET): $(CU_SOURCES) $(CPP_SOURCES) solution.h
	OMPI_CXX=$(NVCC) $(MPICXX) -x cu $(CUDA_FLAGS) --cuda-gpu-arch=$(CUDA_ARCH) $(CU_SOURCES) $(CPP_SOURCES) -lcudart -o $(TARGET) 2>/dev/null || \
	$(NVCC) $(NVCC_FLAGS) -ccbin $(MPICXX) $(CU_SOURCES) $(CPP_SOURCES) -lmpi -o $(TARGET)

# 运行（3 个进程：2 计算节点 + 1 数据服务器）
run: $(TARGET)
	mpirun -np 3 ./$(TARGET)

# 运行（更多进程）
run4: $(TARGET)
	mpirun -np 4 ./$(TARGET)

# 清理
clean:
	rm -f $(TARGET) *.o

# 帮助
help:
	@echo "MPI + CUDA Stencil Computation"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the program"
	@echo "  run     - Run with 3 MPI processes"
	@echo "  run4    - Run with 4 MPI processes"
	@echo "  clean   - Remove build files"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA Toolkit"
	@echo "  - MPI (OpenMPI or MPICH)"
	@echo ""
	@echo "Install MPI:"
	@echo "  Ubuntu/Debian: sudo apt install openmpi-bin libopenmpi-dev"

.PHONY: all run run4 clean help

