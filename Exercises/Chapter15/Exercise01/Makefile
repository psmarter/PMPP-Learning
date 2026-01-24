# 编译器和标志
NVCC = nvcc

# CUDA 架构（根据你的 GPU 调整）
# RTX 40 系列: sm_89
# RTX 30 系列: sm_86
# RTX 20 系列: sm_75
CUDA_ARCH = sm_89
CUDA_FLAGS = -arch=$(CUDA_ARCH) -O3
CXX_FLAGS = -std=c++17

# Common 目录
COMMON_DIR = ../../../../Common

# 包含目录
INCLUDES = -I./include -I$(COMMON_DIR)

# 目录
SRC_DIR = src
OBJ_DIR = obj

# 目标可执行文件
TARGET = test_bfs

# 源文件
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)

# 目标文件
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))

# 默认目标
all: dirs $(TARGET)
	@echo "Build completed successfully!"
	@echo "Run 'make run' to execute the program"

# 创建目录
dirs:
	@mkdir -p $(OBJ_DIR)

# 链接目标文件
$(TARGET): $(OBJ_FILES)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^

# 编译 CUDA 源文件
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# 运行程序
run: $(TARGET)
	./$(TARGET)

# 清理生成的文件
clean:
	rm -rf $(OBJ_DIR) $(TARGET)
	@echo "Cleaned all build files"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  make        - Build the project"
	@echo "  make run    - Build and run the program"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"

.PHONY: all dirs run clean help
