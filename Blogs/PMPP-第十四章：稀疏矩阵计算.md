---
title: PMPP-第十四章：稀疏矩阵计算
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 稀疏矩阵
  - SpMV
  - CSR格式
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: 7af84cf7
date: 2026-01-19 10:49:30
---

## 前言

前几章学习了归约、排序等处理"规则数据"的并行算法。本章学习**稀疏矩阵（Sparse Matrix）**——大部分元素为零的矩阵。稀疏矩阵在科学计算、图算法、机器学习中广泛应用。存储所有零元素既浪费空间又浪费计算，因此需要特殊的存储格式和算法。第十四章讲解常见的稀疏矩阵格式及其在 GPU 上的高效实现。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 稀疏矩阵基础

### 什么是稀疏矩阵

**稀疏矩阵**：非零元素占比很小的矩阵。

```
密集矩阵 (Dense):
[1, 2, 3, 4]
[5, 6, 7, 8]
[9, 10, 11, 12]

稀疏矩阵 (Sparse):
[0, 0, 3, 0]
[0, 0, 0, 0]
[2, 0, 0, 5]
```

**稀疏度**：零元素占比。通常稀疏度 > 90% 就值得用稀疏格式存储。

### 应用场景

| 领域     | 应用              | 稀疏度  |
| -------- | ----------------- | ------- |
| 图算法   | 邻接矩阵          | > 99%   |
| 物理仿真 | 有限元刚度矩阵    | > 95%   |
| 推荐系统 | 用户-物品评分矩阵 | > 99.9% |
| NLP      | 词-文档矩阵       | > 99%   |
| 深度学习 | 剪枝后的权重矩阵  | 50%-90% |

### 为什么需要特殊格式

**存储效率**：

- 密集格式：n² 个元素
- 稀疏格式：O(nnz) 个元素，nnz = 非零元素数

**计算效率**：

- 密集 SpMV：O(n²) 操作
- 稀疏 SpMV：O(nnz) 操作

对于 10000×10000 矩阵，1% 稀疏度：

- 密集：10⁸ 个元素，400 MB
- 稀疏：10⁶ 个非零元素，~12 MB

## 常见稀疏格式

### COO（Coordinate）格式

最直观的格式：存储每个非零元素的 (行, 列, 值)。

```
矩阵:
[0, 0, 3, 0]
[0, 0, 0, 0]
[2, 0, 0, 5]

COO 表示:
row:    [0, 2, 2]
col:    [2, 0, 3]
value:  [3, 2, 5]
```

**存储空间**：3 × nnz

**优点**：简单，构建方便，支持乱序插入

**缺点**：按行遍历效率低，不适合 SpMV

### CSR（Compressed Sparse Row）格式

最常用的格式：压缩行索引。

```
矩阵:
[0, 0, 3, 0]
[0, 0, 0, 0]
[2, 0, 0, 5]

CSR 表示:
row_ptr:  [0, 1, 1, 3]  // 每行的起始位置
col_idx:  [2, 0, 3]      // 列索引
values:   [3, 2, 5]      // 值
```

**row_ptr 解读**：

- 第 0 行：row_ptr[0] 到 row_ptr[1]，即 [0, 1)，包含 1 个元素
- 第 1 行：row_ptr[1] 到 row_ptr[2]，即 [1, 1)，包含 0 个元素
- 第 2 行：row_ptr[2] 到 row_ptr[3]，即 [1, 3)，包含 2 个元素

**存储空间**：(n + 1) + 2 × nnz

**优点**：按行访问高效，SpMV 友好

**缺点**：插入/删除代价高，负载可能不均衡

### CSC（Compressed Sparse Column）格式

CSR 的转置：压缩列索引。

```
col_ptr:  [0, 1, 1, 2, 3]  // 每列的起始位置
row_idx:  [2, 0, 2]         // 行索引
values:   [2, 3, 5]         // 值
```

**适用场景**：按列访问频繁，如 SpMV 的转置操作。

### ELL（ELLPACK/ITPACK）格式

每行填充到相同长度：

```
矩阵:
[1, 0, 2, 0]
[0, 3, 0, 0]
[4, 5, 6, 0]

ELL 表示（每行最多 3 个非零元素）:
values:       col_idx:
[1, 2, *]     [0, 2, *]
[3, *, *]     [1, *, *]
[4, 5, 6]     [0, 1, 2]
```

- 表示填充值（无效元素）

**优点**：规则访问模式，适合 GPU 向量化

**缺点**：行长度差异大时浪费空间

### Hybrid（HYB）格式

ELL + COO 的组合：

- 大部分元素用 ELL 存储（规则部分）
- 超出 ELL 宽度的元素用 COO 存储（溢出部分）

**适用场景**：行长度分布不均匀的矩阵。

## 稀疏矩阵-向量乘法（SpMV）

### 问题定义

y = A × x，其中 A 是 m × n 稀疏矩阵。

```
y[i] = Σ A[i][j] × x[j]  （j 遍历第 i 行的非零元素）
```

SpMV 是科学计算最常见的操作，是迭代求解器（CG、GMRES）的核心。

### CSR SpMV 串行实现

```c
void spmv_csr_sequential(int m, int *row_ptr, int *col_idx, 
                          float *values, float *x, float *y) {
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}
```

### CSR SpMV 并行实现

**每行一个线程**：

```cuda
__global__ void spmv_csr_scalar(int m, int *row_ptr, int *col_idx,
                                 float *values, float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        float sum = 0;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}
```

**问题**：

1. **负载不均衡**：行长度差异导致线程工作量不同
2. **分支发散**：同一 Warp 的线程循环次数不同
3. **非合并访问**：x[col_idx[j]] 是间接访问，不连续

## CSR SpMV 优化

### 每行一个 Warp

用一个 Warp（32 线程）处理一行，Warp 内规约求和：

```cuda
__global__ void spmv_csr_vector(int m, int *row_ptr, int *col_idx,
                                 float *values, float *x, float *y) {
    int row = blockIdx.x * blockDim.x / 32 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    if (row < m) {
        float sum = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        // Warp 内线程协作遍历行
        for (int j = row_start + lane; j < row_end; j += 32) {
            sum += values[j] * x[col_idx[j]];
        }
        
        // Warp 内规约
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
            y[row] = sum;
        }
    }
}
```

**优势**：

- 负载在 Warp 内均衡
- 无分支发散（所有线程做相同次循环）
- 部分合并（values 连续）

### 自适应策略

根据行长度选择策略：

```cuda
if (row_length < 32) {
    // 每行一个线程组（如 8 线程）
} else if (row_length < 256) {
    // 每行一个 Warp
} else {
    // 每行多个 Warp
}
```

cuSPARSE 库会自动选择最优策略。

## ELL SpMV

### 实现

```cuda
__global__ void spmv_ell(int m, int max_nnz_per_row,
                          int *col_idx, float *values,
                          float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        float sum = 0;
        for (int j = 0; j < max_nnz_per_row; j++) {
            int idx = row + j * m;  // 列主序存储
            int col = col_idx[idx];
            if (col >= 0) {  // 有效元素
                sum += values[idx] * x[col];
            }
        }
        y[row] = sum;
    }
}
```

### 列主序布局

```
原始（行主序）:
row 0: [a, b, c]
row 1: [d, e, *]
row 2: [f, g, h]

列主序:
j=0: [a, d, f]  ← 连续访问！
j=1: [b, e, g]
j=2: [c, *, h]
```

**优势**：同一 Warp 的线程访问连续内存，实现合并访问。

### ELL vs CSR

| 特性     | CSR              | ELL                  |
| -------- | ---------------- | -------------------- |
| 空间     | 2×nnz + (m+1)    | 2×m×max_nnz          |
| 访问合并 | 差（值和列索引） | 好（列主序）         |
| 负载均衡 | 需要额外处理     | 自然均衡（固定循环） |
| 适用矩阵 | 通用             | 行长度相近           |

## JDS（Jagged Diagonal Storage）格式

### 思想

按行长度降序排列行，然后用类似 ELL 的方式存储：

```
原始矩阵:
row 0: [a, b]       (长度 2)
row 1: [c, d, e, f] (长度 4)
row 2: [g]          (长度 1)
row 3: [h, i, j]    (长度 3)

按长度排序后:
row 1: [c, d, e, f]
row 3: [h, i, j, *]
row 0: [a, b, *, *]
row 2: [g, *, *, *]

JDS 存储:
jds_ptr:  [0, 4, 7, 9, 10]  // 每"对角线"的起始位置
col_idx:  [1列索引...]
values:   [c, h, a, g, d, i, b, e, j, f]
perm:     [1, 3, 0, 2]      // 行重排映射
```

### 优势

- 短行集中在后面，减少填充浪费
- 前面的迭代负载均衡更好

### 缺点

- 需要额外的行重排数组
- 结果需要按原顺序写回

## 分块格式

### BSR（Block Sparse Row）

把矩阵分成小块（如 4×4），用 CSR 存储块：

```
原始矩阵 (8×8):
[A, 0, B, 0]     A, B, C, D 是 2×2 块
[0, C, 0, D]

BSR 表示（块大小 2×2）:
block_row_ptr: [0, 2, 4]
block_col_idx: [0, 2, 1, 3]
block_values:  [A的4个值, B的4个值, C的4个值, D的4个值]
```

### 优势

1. **减少索引开销**：每个块只需一个 (行, 列) 索引
2. **提高缓存利用**：块内数据连续
3. **利用密集计算**：块内可以用密集矩阵乘法

### 适用场景

物理仿真中的多自由度系统（每个节点多个自由度），自然形成块结构。

## 格式选择指南

### 决策树

```
                    ┌─────────────────┐
                    │    稀疏矩阵      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ 行长度差异大？   │
                    └────────┬────────┘
                   是        │        否
              ┌──────────────┼──────────────┐
              │              │              │
      ┌───────▼───────┐      │      ┌───────▼───────┐
      │ HYB 或 JDS     │      │      │ ELL           │
      └───────────────┘      │      └───────────────┘
                             │
                    ┌────────▼────────┐
                    │ 有块结构？      │
                    └────────┬────────┘
                   是        │        否
              ┌──────────────┼──────────────┐
              │              │              │
      ┌───────▼───────┐      │      ┌───────▼───────┐
      │ BSR            │      │      │ CSR           │
      └───────────────┘      │      └───────────────┘
```

### 格式对比

| 格式 | 构建难度 | 空间效率 | SpMV 性能 | 适用场景 |
| ---- | -------- | -------- | --------- | -------- |
| COO  | 容易     | 中       | 差        | 构建阶段 |
| CSR  | 中       | 好       | 中        | 通用     |
| ELL  | 中       | 差-中    | 好        | 均匀行长 |
| HYB  | 复杂     | 中       | 好        | 不均匀   |
| BSR  | 复杂     | 好       | 很好      | 块结构   |

## 格式转换

### COO 到 CSR

```cuda
void coo_to_csr(int m, int nnz, int *coo_row, int *coo_col, 
                float *coo_val, int *csr_row_ptr, int *csr_col, 
                float *csr_val) {
    // 1. 统计每行元素数
    for (int i = 0; i < nnz; i++) {
        csr_row_ptr[coo_row[i] + 1]++;
    }
    
    // 2. 前缀和得到行指针
    for (int i = 0; i < m; i++) {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }
    
    // 3. 填充列索引和值
    int *temp = calloc(m, sizeof(int));
    for (int i = 0; i < nnz; i++) {
        int row = coo_row[i];
        int pos = csr_row_ptr[row] + temp[row];
        csr_col[pos] = coo_col[i];
        csr_val[pos] = coo_val[i];
        temp[row]++;
    }
}
```

### 并行转换

利用前面学的技术：

1. **统计**：并行直方图
2. **前缀和**：并行扫描
3. **分配**：原子操作或前缀和确定位置

## cuSPARSE 库

### 基本使用

```cuda
#include <cusparse.h>

void spmv_cusparse(int m, int n, int nnz,
                   int *row_ptr, int *col_idx, float *values,
                   float *x, float *y) {
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // 创建矩阵描述符
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, m, n, nnz,
                      row_ptr, col_idx, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    // 创建向量描述符
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, m, y, CUDA_R_32F);
    
    // 分配临时空间
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                            &bufferSize);
    
    void *buffer;
    cudaMalloc(&buffer, bufferSize);
    
    // 执行 SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY,
                 CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);
    
    // 清理
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaFree(buffer);
    cusparseDestroy(handle);
}
```

### cuSPARSE 特性

| 功能     | 说明                 |
| -------- | -------------------- |
| SpMV     | 稀疏矩阵-向量乘      |
| SpMM     | 稀疏矩阵-密集矩阵乘  |
| SpGEMM   | 稀疏矩阵-稀疏矩阵乘  |
| 三角求解 | 稀疏三角矩阵求解     |
| 格式转换 | COO/CSR/CSC/BSR 互转 |
| 矩阵分析 | 着色、重排序         |

## 性能优化总结

### 内存访问优化

1. **值和列索引**：CSR 中这两者通常连续访问，合并良好
2. **x 向量**：间接访问，考虑用纹理缓存
3. **列主序 ELL**：保证同一 Warp 的线程访问连续

### 负载均衡

1. **分行策略**：短行用少线程，长行用多线程
2. **分块处理**：把非零元素均匀分配给线程块
3. **动态调度**：运行时根据行长度分配资源

### 减少开销

1. **合并迭代**：多次 SpMV 之间不必来回拷贝
2. **重用分析结果**：矩阵结构不变时，analysis 只做一次
3. **混合精度**：索引用 int32，值用 fp16/bf16

## 小结

第十四章深入讲解稀疏矩阵：

**存储格式**：COO（简单）、CSR（通用）、ELL（规则）、BSR（块结构）。选择取决于矩阵特性和操作类型。

**CSR SpMV**：最常用。每行一线程简单但负载不均；每行一 Warp 更均衡但需要规约。自适应策略根据行长度选择。

**ELL SpMV**：列主序存储保证合并访问。适合行长度相近的矩阵。

**格式转换**：COO → CSR 用前缀和确定位置。并行转换利用直方图和扫描。

**cuSPARSE**：生产环境首选。提供多种格式和操作，自动选择最优算法。

稀疏矩阵是科学计算的基础。掌握格式选择和 SpMV 优化，就能高效处理图算法、物理仿真、机器学习中的大规模稀疏数据。

---

## 🚀 下一步

---

## 📚 参考资料

- PMPP 第四版 Chapter 14
- [第十四章：稀疏矩阵计算](https://smarter.xin/posts/7af84cf7/)

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
