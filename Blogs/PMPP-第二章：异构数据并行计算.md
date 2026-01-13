---
title: PMPP-第二章：异构数据并行计算
date: 2026-01-11 19:38:29
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 向量加法
categories: 知识分享
cover: /img/PMPP.jpg
---

## 前言

第一章讲了理论，第二章开始写代码了。虽然例子是经典的向量加法，但它包含了CUDA编程的所有核心环节：内存管理、kernel编写、线程组织。掌握这个简单例子，后面的复杂应用就是在此基础上的扩展。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 为什么从向量加法开始

### 数据并行的典型例子

向量加法是数据并行的最佳入口：

```
C[0] = A[0] + B[0]
C[1] = A[1] + B[1]
...
C[n-1] = A[n-1] + B[n-1]
```

每个元素的计算完全独立，C[0]不需要等C[1]算完。这种独立性正是并行计算的黄金场景。

### 内存受限问题

向量加法的算术强度很低：

- 每元素：读2个float + 写1个float = 12字节
- 计算：1次浮点加法
- 算术强度：1 FLOP / 12 Bytes ≈ 0.083 FLOP/Byte

典型内存受限（Memory-Bound）问题。GPU计算单元会经常等数据。虽然性能达不到峰值，但作为入门例子足够简单直观。

## CUDA程序结构：三步走

### CPU版本（对比）

```c
void vecAddCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}
```

串行执行，n=10000就要循环10000次。

### CUDA版本

#### 1. Host端准备

```cuda
int n = 10000;
size_t size = n * sizeof(float);

// 分配Host内存
float *h_A = (float*)malloc(size);
float *h_B = (float*)malloc(size);
float *h_C = (float*)malloc(size);

// 初始化数据
for (int i = 0; i < n; i++) {
    h_A[i] = i * 1.0f;
    h_B[i] = i * 2.0f;
}
```

`h_`前缀表示Host变量，这是个好习惯。

#### 2. Device端准备

```cuda
// 分配Device内存
float *d_A, *d_B, *d_C;
cudaMalloc((void**)&d_A, size);
cudaMalloc((void**)&d_B, size);
cudaMalloc((void**)&d_C, size);

// Host → Device
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
```

**关键**：

- `cudaMalloc`参数是二级指针（需要修改指针值）
- `d_A`是Device指针，在Host代码中不能直接解引用`d_A[0]`（会段错误）

#### 3. 执行与回传

```cuda
// 启动kernel
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

// Device → Host
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

// 验证
for (int i = 0; i < n; i++) {
    if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
        printf("Error at index %d\n", i);
    }
}

// 清理
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
free(h_A);
free(h_B);
free(h_C);
```

## Kernel函数

### 基本结构

```cuda
__global__ void vecAdd(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

**逐行解析**：

- `__global__`：GPU上执行，CPU调用
  - `__device__`：GPU上执行，GPU调用
  - `__host__`：CPU上执行，CPU调用（默认，可省略）

- 线程索引计算：`i = blockIdx.x * blockDim.x + threadIdx.x`
  - `blockIdx.x`：block在grid中的索引
  - `blockDim.x`：block的大小
  - `threadIdx.x`：thread在block中的索引

- 边界检查：`if (i < n)` 必须有（总线程数通常多于数组元素）

### 线程层次结构

```
Grid
├── Block 0 (256 threads)
│   ├── Thread 0   → i = 0*256 + 0 = 0
│   ├── Thread 1   → i = 0*256 + 1 = 1
│   └── Thread 255 → i = 0*256 + 255 = 255
├── Block 1 (256 threads)
│   ├── Thread 0   → i = 1*256 + 0 = 256
│   └── Thread 255 → i = 1*256 + 255 = 511
└── ...
```

每个thread得到唯一索引，对应数组元素。

### 为什么256个threads？

不是随便选的：

1. **Warp的倍数**：GPU以32线程为一组（warp）执行，256 = 8 × 32
2. **硬件限制**：每block最多1024 threads
3. **经验值**：128-512通常性能较好

具体最优值需要profiling确定。

### 边界检查的必要性

```cuda
blocksPerGrid = (10000 + 255) / 256 = 40
总threads = 40 × 256 = 10240
```

多出240个线程。不检查边界会越界访问，导致错误或崩溃。

## 内存管理

### Host vs Device内存

**关键**：两个独立的内存空间，不能直接互访。

- **Host Memory**：CPU的DDR4/DDR5
- **Device Memory**：GPU的GDDR6/HBM

**错误示例**：

```cuda
float *d_A;
cudaMalloc((void**)&d_A, size);
d_A[0] = 1.0f;  // 段错误！CPU不能直接访问GPU内存
```

**正确做法**：

```cuda
float *h_A = (float*)malloc(size);
h_A[0] = 1.0f;
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
```

### 数据传输开销

PCIe带宽（~32 GB/s）远低于GPU内存带宽（500+ GB/s）。对于简单计算，传输时间可能是计算时间的数十倍。

**优化原则**：

- 减少传输次数（批量传输）
- 保持数据在GPU（多步计算不回传）
- 异步传输与计算重叠（高级技巧）

### Unified Memory（可选）

从CUDA 6.0起可以用：

```cuda
float *data;
cudaMallocManaged(&data, size);

data[0] = 1.0f;              // CPU访问
kernel<<<...>>>(data);       // GPU访问（自动迁移）
printf("%f\n", data[0]);     // CPU访问（自动传回）

cudaFree(data);
```

方便，但有性能开销。学习原型开发友好，生产环境建议显式管理。

## 执行配置

### 启动语法

```cuda
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

完整形式：

```cuda
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args);
```

- `gridDim`：grid的维度（1D/2D/3D）
- `blockDim`：block的维度
- `sharedMem`：共享内存大小（可选，默认0）
- `stream`：CUDA流（可选，默认0）

### 计算grid大小

```cuda
int threads = 256;
int blocks = (n + threads - 1) / threads;  // 向上取整
```

数学等价于`ceil(n / threads)`，但整数运算更高效。

## 错误处理

CUDA函数返回`cudaError_t`，需要显式检查：

```cuda
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 使用
CUDA_CHECK(cudaMalloc((void**)&d_A, size));
CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
```

Kernel启动不返回错误码，需要：

```cuda
vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, n);
CUDA_CHECK(cudaGetLastError());          // 检查启动错误
CUDA_CHECK(cudaDeviceSynchronize());     // 同步并检查执行错误
```

## 小结

第二章通过向量加法建立了CUDA编程的基本框架：

**核心流程**：内存分配 → 数据传输 → kernel启动 → 结果回传，这是所有CUDA程序的骨架。

**线程组织**：Grid/Block/Thread三级结构，索引计算`i = blockIdx.x * blockDim.x + threadIdx.x`要烂熟于心。

**内存模型**：Host和Device是独立空间，必须显式传输。数据传输开销不容忽视。

**性能认知**：向量加法虽然能在GPU上跑，但受内存带宽限制，性能提升有限。真正发挥GPU优势需要高算术强度的任务。

**代码习惯**：

- 变量命名区分h_/d_
- 边界检查必须严格
- 错误处理不能省略

下一章进入多维数据处理（矩阵、图像），会用到2D Grid/Block组织。理解了一维的原理，多维只是自然扩展。

---

**参考资料：**

- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
