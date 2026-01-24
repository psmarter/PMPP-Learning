# 第二十章 Exercise01 Makefile
# MPI + CUDA 分布式模板计算

NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_60 -std=c++17

# MPI 路径（适用于 Ubuntu/Debian 的 OpenMPI）
MPI_INCLUDE = /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB = /usr/lib/x86_64-linux-gnu/openmpi/lib

# 检查 MPI 头文件是否存在
MPI_AVAILABLE := $(shell test -f $(MPI_INCLUDE)/mpi.h && echo yes || echo no)

ifeq ($(MPI_AVAILABLE),no)
$(warning MPI not found at $(MPI_INCLUDE))
$(warning Please install OpenMPI: sudo apt install openmpi-bin libopenmpi-dev)
endif

# MPI 编译标志
MPI_FLAGS = -I$(MPI_INCLUDE) -I$(MPI_INCLUDE)/openmpi -L$(MPI_LIB) -lmpi

# 目标可执行文件
TARGET = stencil_mpi

# 源文件
SOURCES = solution.cu test.cpp

# 默认目标
all: $(TARGET)

# 编译
$(TARGET): $(SOURCES) solution.h
	$(NVCC) $(CUDA_FLAGS) $(SOURCES) $(MPI_FLAGS) -o $(TARGET)

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

