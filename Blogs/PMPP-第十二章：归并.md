---
title: PMPP-第十二章：归并
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 归并
  - Merge
  - 动态数据识别
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: '31928809'
date: 2026-01-18 21:42:06
---

## 前言

第十一章学习了前缀和，本章学习**归并（Merge）**——把两个有序数组合并成一个有序数组。归并是排序算法的核心组件（归并排序），也是数据库操作的基础（JOIN 操作）。并行归并的挑战在于：**输出位置取决于两个数组的数据内容**，不像矩阵乘法那样可以根据线程索引直接确定位置。第十二章讲解如何使用**协同排名（Co-Rank）**技术解决这个"动态数据识别"问题。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 归并基础

### 什么是归并

**归并**：给定两个**已排序**的数组 A 和 B，生成一个包含所有元素的**有序**数组 C。

```
A = [1, 3, 5, 7, 9]
B = [2, 4, 6, 8, 10]
C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### 串行归并

经典的双指针算法：

```c
void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    
    // 处理剩余元素
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}
```

时间复杂度 O(m+n)，空间复杂度 O(1)（不计输出）。

### 并行化的挑战

**问题**：输出位置 k 对应的输入位置 (i, j) 取决于数据内容。

```
要填充 C[5]，需要知道：
- A 和 B 中有多少元素 ≤ C[5] 的值
- 这取决于 A 和 B 的具体内容
```

不像矩阵乘法那样可以根据线程 ID 直接计算输入位置。

**核心问题**：如何让每个线程知道自己应该处理 A 和 B 的哪一段？

## 协同排名（Co-Rank）

### 核心洞察

对于输出位置 k，假设它对应 A 中的前 i 个元素和 B 中的前 j 个元素，则：

```
i + j = k
```

这就是**协同排名**约束。

**问题转化**：给定 k，找到满足约束且保持有序性的 (i, j)。

### 二分搜索解法

**有序性条件**：

```
A[i-1] ≤ B[j]  且  B[j-1] ≤ A[i]
```

即：A 中第 i 个元素应该排在 B 的第 j 个元素之前或相等，反之亦然。

**算法**：

```c
int co_rank(int k, int *A, int m, int *B, int n) {
    // i 的范围
    int i_low = max(0, k - n);
    int i_high = min(k, m);
    
    while (i_low < i_high) {
        int i = (i_low + i_high) / 2;
        int j = k - i;
        
        if (i > 0 && j < n && A[i-1] > B[j]) {
            // A[i-1] 太大，i 需要减小
            i_high = i;
        } else if (j > 0 && i < m && B[j-1] > A[i]) {
            // B[j-1] 太大，i 需要增大
            i_low = i + 1;
        } else {
            // 找到了
            return i;
        }
    }
    return i_low;
}
```

**时间复杂度**：O(log(min(k, m, n)))

### 示例

```
A = [1, 3, 5, 7, 9]  (m=5)
B = [2, 4, 6, 8, 10] (n=5)
k = 6（输出位置 6）

i + j = 6

尝试 i=3: j=3
  A[2]=5 ≤ B[3]=8? ✓
  B[2]=6 ≤ A[3]=7? ✓
  找到！

所以 C[0..5] 来自 A[0..2] 和 B[0..2]
C[6..9] 来自 A[3..4] 和 B[3..4]
```

## 基础并行归并

### 每线程一个元素

最简单的并行化：每个线程负责计算一个输出元素。

```cuda
__global__ void merge_basic(int *A, int m, int *B, int n, int *C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < m + n) {
        int i = co_rank(k, A, m, B, n);
        int j = k - i;
        
        int i_next = co_rank(k + 1, A, m, B, n);
        int j_next = k + 1 - i_next;
        
        // 确定这个位置的值
        if (i_next > i) {
            C[k] = A[i];
        } else {
            C[k] = B[j];
        }
    }
}
```

### 问题

**效率低**：每个线程都做一次二分搜索（O(log n)），总工作量 O(n log n)。

串行只需 O(n)，并行反而做了更多工作！

## 分块并行归并

### 思路

不要每个元素都二分搜索，而是：

1. 把输出分成若干块（Tile）
2. 每个 Tile 做一次 Co-Rank 找到边界
3. Tile 内部用串行归并

### 算法

```
输出大小：m + n
Tile 大小：T
Tile 数量：(m + n + T - 1) / T

对于 Tile k：
  起始位置：k * T
  结束位置：min((k+1) * T, m+n)
  
  Co-Rank 找到 (i_start, j_start) 和 (i_end, j_end)
  
  归并 A[i_start..i_end] 和 B[j_start..j_end]
```

### CUDA 实现

```cuda
#define TILE_SIZE 1024

__global__ void merge_tiled(int *A, int m, int *B, int n, int *C) {
    // 每个 Block 处理一个 Tile
    int k_start = blockIdx.x * TILE_SIZE;
    int k_end = min(k_start + TILE_SIZE, m + n);
    
    // Co-Rank 找边界
    __shared__ int i_start, j_start, i_end, j_end;
    
    if (threadIdx.x == 0) {
        i_start = co_rank(k_start, A, m, B, n);
        j_start = k_start - i_start;
        i_end = co_rank(k_end, A, m, B, n);
        j_end = k_end - i_end;
    }
    __syncthreads();
    
    // 每个线程处理 Tile 中的一部分
    int tile_size = k_end - k_start;
    int elements_per_thread = (tile_size + blockDim.x - 1) / blockDim.x;
    
    int local_k = threadIdx.x * elements_per_thread;
    int local_k_end = min(local_k + elements_per_thread, tile_size);
    
    // 线程内 Co-Rank
    int A_seg_size = i_end - i_start;
    int B_seg_size = j_end - j_start;
    
    int local_i = co_rank(local_k, A + i_start, A_seg_size, 
                                   B + j_start, B_seg_size);
    int local_j = local_k - local_i;
    
    int local_i_end = co_rank(local_k_end, A + i_start, A_seg_size,
                                           B + j_start, B_seg_size);
    int local_j_end = local_k_end - local_i_end;
    
    // 串行归并
    merge_sequential(A + i_start + local_i, local_i_end - local_i,
                     B + j_start + local_j, local_j_end - local_j,
                     C + k_start + local_k);
}
```

### 工作量分析

| 操作          | 次数               | 单次复杂度    |
| ------------- | ------------------ | ------------- |
| Block Co-Rank | (m+n)/T            | O(log(m+n))   |
| 线程 Co-Rank  | (m+n)/T × blockDim | O(log T)      |
| 串行归并      | (m+n)/T × blockDim | O(T/blockDim) |

总工作量 ≈ O(m+n)，接近串行！

## 共享内存优化

### 动机

前面的实现每个线程都访问全局内存做串行归并。如果把 Tile 数据加载到共享内存，可以大幅减少全局内存访问。

### 实现

```cuda
#define TILE_SIZE 1024

__global__ void merge_shared(int *A, int m, int *B, int n, int *C) {
    __shared__ int A_s[TILE_SIZE];
    __shared__ int B_s[TILE_SIZE];
    __shared__ int C_s[TILE_SIZE];
    
    int k_start = blockIdx.x * TILE_SIZE;
    int k_end = min(k_start + TILE_SIZE, m + n);
    int tile_size = k_end - k_start;
    
    // Co-Rank 找边界
    __shared__ int i_start, j_start, i_end, j_end;
    
    if (threadIdx.x == 0) {
        i_start = co_rank(k_start, A, m, B, n);
        j_start = k_start - i_start;
        i_end = co_rank(k_end, A, m, B, n);
        j_end = k_end - i_end;
    }
    __syncthreads();
    
    int A_size = i_end - i_start;
    int B_size = j_end - j_start;
    
    // 加载到共享内存
    for (int i = threadIdx.x; i < A_size; i += blockDim.x) {
        A_s[i] = A[i_start + i];
    }
    for (int i = threadIdx.x; i < B_size; i += blockDim.x) {
        B_s[i] = B[j_start + i];
    }
    __syncthreads();
    
    // 线程内 Co-Rank（使用共享内存）
    int elements_per_thread = (tile_size + blockDim.x - 1) / blockDim.x;
    int local_k = threadIdx.x * elements_per_thread;
    int local_k_end = min(local_k + elements_per_thread, tile_size);
    
    int local_i = co_rank(local_k, A_s, A_size, B_s, B_size);
    int local_j = local_k - local_i;
    
    int local_i_end = co_rank(local_k_end, A_s, A_size, B_s, B_size);
    int local_j_end = local_k_end - local_i_end;
    
    // 归并到共享内存
    merge_sequential(A_s + local_i, local_i_end - local_i,
                     B_s + local_j, local_j_end - local_j,
                     C_s + local_k);
    __syncthreads();
    
    // 写回全局内存
    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
        C[k_start + i] = C_s[i];
    }
}
```

### 性能提升

| 版本     | 全局内存访问 | 性能 |
| -------- | ------------ | ---- |
| 基础     | O(m+n) 次    | 1×   |
| 分块     | O(m+n) 次    | ~5×  |
| 共享内存 | O(m+n)/T 次  | ~10× |

## 循环缓冲区优化

### 问题

共享内存有限，如果 A_size + B_size > 共享内存容量怎么办？

### 思路：流式处理

用**循环缓冲区**逐块处理：

1. 加载 A 和 B 的一小块到共享内存
2. 归并能归并的部分
3. 加载下一块，继续归并
4. 重复直到完成

### 关键点

**消费跟踪**：记录 A 和 B 各消费了多少元素。

**缓冲区管理**：循环使用缓冲区空间。

```cuda
__global__ void merge_circular(int *A, int m, int *B, int n, int *C) {
    __shared__ int buffer_A[BUFFER_SIZE];
    __shared__ int buffer_B[BUFFER_SIZE];
    
    // ... 初始化 ...
    
    int a_consumed = 0, b_consumed = 0;
    int a_loaded = 0, b_loaded = 0;
    int c_produced = 0;
    
    while (c_produced < tile_size) {
        // 填充缓冲区
        while (a_loaded - a_consumed < BUFFER_SIZE && a_loaded < A_size) {
            buffer_A[a_loaded % BUFFER_SIZE] = A_s[a_loaded];
            a_loaded++;
        }
        while (b_loaded - b_consumed < BUFFER_SIZE && b_loaded < B_size) {
            buffer_B[b_loaded % BUFFER_SIZE] = B_s[b_loaded];
            b_loaded++;
        }
        
        // 归并一批
        // ...
    }
}
```

这种技术在数据量远超共享内存时很有用。

## 归并路径可视化

### Co-Rank 的几何解释

把归并过程可视化为 2D 网格中的路径：

```
    B: 0   1   2   3   4
       +---+---+---+---+
  A:0  |   |   |   |   |
       +---+---+---+---+
    1  |   |   |   |   |
       +---+---+───+───+
    2  |   |   |\  |   |
       +---+---+─\─+---+
    3  |   |   |  \|   |
       +---+---+---+\--+
    4  |   |   |   | \ |
       +---+---+---+---+
```

**路径规则**：

- 从 (0,0) 到 (m,n)
- 每步向右（取 A 元素）或向下（取 B 元素）
- 选择较小的元素决定方向

**Co-Rank(k)**：路径上第 k 步的位置。

### 并行化视角

```
把输出分成 T 段：
  段 0: 路径 [0, T)
  段 1: 路径 [T, 2T)
  ...

每段的起点由 Co-Rank 确定。
```

## 归并排序

### 分治结构

归并排序 = 递归拆分 + 归并合并

```
[3,1,7,4,5,2,6,8]
       ↓ 拆分
[3,1,7,4] [5,2,6,8]
    ↓ 拆分
[3,1] [7,4] [5,2] [6,8]
  ↓ 拆分
[3][1] [7][4] [5][2] [6][8]
  ↓ 归并
[1,3] [4,7] [2,5] [6,8]
    ↓ 归并
[1,3,4,7] [2,5,6,8]
       ↓ 归并
[1,2,3,4,5,6,7,8]
```

### 并行归并排序

每层归并可以并行：

```cuda
void parallel_merge_sort(int *data, int n) {
    // 从单元素开始，逐层归并
    for (int width = 1; width < n; width *= 2) {
        int num_merges = (n + 2 * width - 1) / (2 * width);
        
        // 并行执行所有归并
        merge_kernel<<<num_merges, BLOCK_SIZE>>>(
            data, n, width
        );
    }
}

__global__ void merge_kernel(int *data, int n, int width) {
    int merge_id = blockIdx.x;
    int left = merge_id * 2 * width;
    int mid = min(left + width, n);
    int right = min(left + 2 * width, n);
    
    // 归并 [left, mid) 和 [mid, right)
    merge_shared(data + left, mid - left, 
                 data + mid, right - mid, 
                 temp + left);
}
```

### 复杂度

| 指标     | 串行归并排序 | 并行归并排序 |
| -------- | ------------ | ------------ |
| 层数     | log₂N        | log₂N        |
| 每层工作 | O(N)         | O(N)         |
| 总工作   | O(N log N)   | O(N log N)   |
| 并行时间 | O(N log N)   | O(log²N)*    |

*假设有足够多的处理器，每层归并并行完成。

## 与其他算法的关系

### 归并 vs 基数排序

| 特性       | 归并排序   | 基数排序    |
| ---------- | ---------- | ----------- |
| 比较次数   | O(N log N) | O(N × 位数) |
| 稳定性     | 稳定       | 稳定        |
| 适用类型   | 通用       | 整数/定长键 |
| GPU 友好度 | 中等       | 高          |

基数排序在 GPU 上通常更快，但归并排序适用范围更广。

### 归并 vs 快速排序

| 特性       | 归并排序   | 快速排序 |
| ---------- | ---------- | -------- |
| 最坏复杂度 | O(N log N) | O(N²)    |
| 空间       | O(N)       | O(log N) |
| 并行化     | 容易       | 较难     |
| 稳定性     | 稳定       | 不稳定   |

归并排序的确定性和稳定性使其在并行计算中更受欢迎。

## CUB/Thrust 库

### 使用 Thrust

```cuda
#include <thrust/merge.h>
#include <thrust/device_vector.h>

void mergeWithThrust(int *A, int m, int *B, int n, int *C) {
    thrust::device_ptr<int> d_A(A);
    thrust::device_ptr<int> d_B(B);
    thrust::device_ptr<int> d_C(C);
    
    thrust::merge(d_A, d_A + m, d_B, d_B + n, d_C);
}
```

### 使用 CUB

```cuda
#include <cub/cub.cuh>

void mergeWithCub(int *d_A, int m, int *d_B, int n, int *d_C) {
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    
    cub::DeviceMerge::Merge(d_temp, temp_bytes, 
                            d_A, m, d_B, n, d_C);
    
    cudaMalloc(&d_temp, temp_bytes);
    
    cub::DeviceMerge::Merge(d_temp, temp_bytes,
                            d_A, m, d_B, n, d_C);
    
    cudaFree(d_temp);
}
```

## 小结

第十二章深入讲解并行归并：

**核心挑战**：输出位置取决于输入数据内容（动态数据识别），不能简单根据线程 ID 确定。

**Co-Rank 技术**：给定输出位置 k，用二分搜索找到对应的输入位置 (i, j)，满足 i + j = k。这是并行归并的关键。

**分块归并**：把输出分成 Tile，每个 Tile 做一次 Co-Rank 找边界，Tile 内串行归并。平衡了并行开销和工作效率。

**共享内存优化**：把 Tile 数据加载到共享内存，减少全局内存访问。循环缓冲区处理大 Tile。

**归并路径**：归并过程可视化为 2D 网格中的路径，Co-Rank 找的是路径上的点。

**归并排序**：log N 层归并，每层可以并行。总工作量 O(N log N)，并行时间可达 O(log² N)。

归并是排序、数据库操作的基础。掌握 Co-Rank 技术，就能处理各种"输出位置依赖输入数据"的问题。下一章学习排序——归并的直接应用。

---

## 🚀 下一步

---

## 📚 参考资料

- PMPP 第四版 Chapter 12
- [第十二章：归并](https://smarter.xin/posts/31928809/)

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
