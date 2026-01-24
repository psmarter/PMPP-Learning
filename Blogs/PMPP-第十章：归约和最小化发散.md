---
title: PMPP-第十章：归约和最小化发散
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 归约
  - 分支发散
  - 并行算法
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: 43b40d12
date: 2026-01-18 10:14:18
---

## 前言

第九章学习了直方图计算，使用原子操作处理输出冲突。本章学习**归约（Reduction）**——把一组数据"归约"成一个值，例如求和、求最大值。归约操作看似简单，实际涉及并行算法的核心问题：如何高效地合并部分结果？如何避免分支发散？如何最大化硬件利用率？第十章系统讲解这些优化技术。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 归约基础

### 什么是归约

**归约**：用一个**二元结合运算**把 N 个元素合并成 1 个值。

**常见归约操作**：

| 操作   | 运算符 | 单位元 | 示例                      |
| ------ | ------ | ------ | ------------------------- |
| 求和   | +      | 0      | 1+2+3+4 = 10              |
| 求积   | ×      | 1      | 1×2×3×4 = 24              |
| 最大值 | max    | -∞     | max(1,5,3,2) = 5          |
| 最小值 | min    | +∞     | min(1,5,3,2) = 1          |
| 逻辑与 | &&     | true   | true && false = false     |
| 逻辑或 | \|\|   | false  | true \|\| false = true    |
| 位与   | &      | ~0     | 0b1100 & 0b1010 = 0b1000  |
| 位或   | \|     | 0      | 0b1100 \| 0b1010 = 0b1110 |

**结合律**是关键：`(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`

有了结合律，就能任意分组并行计算。

### 串行归约

```c
float sum_sequential(float *data, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}
```

时间复杂度 O(n)，无法利用并行性。

### 并行归约的思路

**树形归约**：

```
Level 0: [a0, a1, a2, a3, a4, a5, a6, a7]
Level 1: [a0+a1, a2+a3, a4+a5, a6+a7]
Level 2: [a0+a1+a2+a3, a4+a5+a6+a7]
Level 3: [a0+a1+a2+a3+a4+a5+a6+a7]
```

每层把元素数减半，log₂N 层后得到结果。

**时间复杂度**：O(log N) 步，每步 O(N/2^k) 次操作

**工作量**：总操作数 = N/2 + N/4 + ... + 1 = N - 1（与串行相同）

**并行度**：第 k 层需要 N/2^k 个并行操作

## 朴素并行归约

### 相邻配对

最直观的并行化：相邻元素配对求和

```cuda
__global__ void reduce_naive(float *g_data, float *g_result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载到共享内存
    sdata[tid] = (i < n) ? g_data[i] : 0;
    __syncthreads();
    
    // 树形归约
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // 写回结果
    if (tid == 0) {
        g_result[blockIdx.x] = sdata[0];
    }
}
```

### 问题：分支发散

看这个条件：`if (tid % (2 * stride) == 0)`

**Level 0** (stride=1)：线程 0,2,4,6... 活跃，1,3,5,7... 空闲
**Level 1** (stride=2)：线程 0,4,8... 活跃
**Level 2** (stride=4)：线程 0,8... 活跃

Warp 内有的线程活跃、有的空闲 = **分支发散**！

Warp 必须串行执行两个分支，效率减半。

### 问题：Bank 冲突

`sdata[tid]` 和 `sdata[tid + stride]` 的访问模式：

- stride=1：线程 0 访问 [0,1]，线程 2 访问 [2,3]...（无冲突）
- stride=16：线程 0 访问 [0,16]，线程 32 访问 [32,48]...

stride 是 2 的幂时，可能产生 Bank 冲突。

## 优化 1：交错配对

### 改进思路

把"空闲线程在右边"改成"空闲线程在后面"：

```
朴素：线程 0,2,4,6 活跃，1,3,5,7 空闲
交错：线程 0,1,2,3 活跃，4,5,6,7 空闲
```

这样，活跃线程是连续的，前几个 Warp 满载，最后的 Warp 才逐渐空闲。

### 实现

```cuda
__global__ void reduce_interleaved(float *g_data, float *g_result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_data[i] : 0;
    __syncthreads();
    
    // 交错归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_result[blockIdx.x] = sdata[0];
    }
}
```

### 分析

**Level 0** (stride=128)：线程 0-127 活跃
**Level 1** (stride=64)：线程 0-63 活跃
**Level 2** (stride=32)：线程 0-31 活跃（恰好一个 Warp）

前几层有多个完整 Warp 同时工作，分支发散只在最后几层出现。

**性能提升**：约 2× 相比朴素版本。

## 优化 2：首次加载时归约

### 观察

每个线程只加载一个元素，但 Block 数可能远多于 SM 数。如果让每个线程加载多个元素并在加载时求和，可以减少 Block 数，提高效率。

### 实现

```cuda
__global__ void reduce_first_add(float *g_data, float *g_result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 首次加载时就求和两个元素
    float sum = 0;
    if (i < n) sum += g_data[i];
    if (i + blockDim.x < n) sum += g_data[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // 后续归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_result[blockIdx.x] = sdata[0];
    }
}
```

### 扩展：Grid-Stride 加载

让每个线程加载多个元素：

```cuda
__global__ void reduce_grid_stride(float *g_data, float *g_result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    // Grid-stride 累加
    float sum = 0;
    while (i < n) {
        sum += g_data[i];
        i += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // 树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_result[blockIdx.x] = sdata[0];
    }
}
```

**优势**：

1. Block 数量可以固定（如 256），不随数据量变化
2. 每线程做更多有效工作，分摊同步开销
3. 更好的指令级并行

## 优化 3：展开最后 Warp

### 观察

当 stride < 32 时，只有一个 Warp 在工作。Warp 内线程自动同步（SIMT），不需要 `__syncthreads()`！

### 实现

```cuda
__device__ void warpReduce(volatile float *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_warp_unroll(float *g_data, float *g_result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    sdata[tid] = 0;
    if (i < n) sdata[tid] += g_data[i];
    if (i + blockDim.x < n) sdata[tid] += g_data[i + blockDim.x];
    __syncthreads();
    
    // 常规归约（stride >= 64）
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp 内展开（stride < 32）
    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    
    if (tid == 0) {
        g_result[blockIdx.x] = sdata[0];
    }
}
```

### 关键点

**volatile**：告诉编译器不要优化掉对 sdata 的读写，确保每次都访问共享内存。

**隐式同步**：Warp 内的 32 个线程执行相同指令，自动保持同步。

**性能提升**：减少 5 次 `__syncthreads()` 调用。

## 优化 4：完全展开

### 当 Block 大小已知时

如果 Block 大小是编译时常量（如 256），可以完全展开循环：

```cuda
template <unsigned int blockSize>
__global__ void reduce_complete_unroll(float *g_data, float *g_result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    
    sdata[tid] = 0;
    if (i < n) sdata[tid] += g_data[i];
    if (i + blockSize < n) sdata[tid] += g_data[i + blockSize];
    __syncthreads();
    
    // 编译时展开
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Warp 内展开
    if (tid < 32) {
        volatile float *smem = sdata;
        if (blockSize >= 64) smem[tid] += smem[tid + 32];
        if (blockSize >= 32) smem[tid] += smem[tid + 16];
        if (blockSize >= 16) smem[tid] += smem[tid + 8];
        if (blockSize >= 8)  smem[tid] += smem[tid + 4];
        if (blockSize >= 4)  smem[tid] += smem[tid + 2];
        if (blockSize >= 2)  smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        g_result[blockIdx.x] = sdata[0];
    }
}
```

### 启动

```cuda
// 根据 Block 大小选择模板实例
switch (blockSize) {
    case 512: reduce_complete_unroll<512><<<grid, 512, 512*sizeof(float)>>>(...); break;
    case 256: reduce_complete_unroll<256><<<grid, 256, 256*sizeof(float)>>>(...); break;
    case 128: reduce_complete_unroll<128><<<grid, 128, 128*sizeof(float)>>>(...); break;
    // ...
}
```

**优势**：编译器可以完全展开循环，消除循环开销和分支。

## 优化 5：Warp Shuffle

### 现代方法

从 Kepler 架构开始，CUDA 提供 **Warp Shuffle** 指令，线程可以直接交换寄存器值，无需共享内存：

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 完整实现

```cuda
__global__ void reduce_shuffle(float *g_data, float *g_result, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    
    // Grid-stride 累加
    float sum = 0;
    while (i < n) {
        sum += g_data[i];
        i += gridSize;
    }
    
    // Warp 内归约（shuffle）
    sum = warpReduceSum(sum);
    
    // Warp 间归约
    __shared__ float warpSums[32];  // 最多 32 个 Warp
    int lane = tid % 32;
    int wid = tid / 32;
    
    if (lane == 0) {
        warpSums[wid] = sum;
    }
    __syncthreads();
    
    // 第一个 Warp 做最终归约
    if (wid == 0) {
        sum = (tid < blockDim.x / 32) ? warpSums[lane] : 0;
        sum = warpReduceSum(sum);
        
        if (tid == 0) {
            g_result[blockIdx.x] = sum;
        }
    }
}
```

### Shuffle 函数

| 函数               | 功能                    |
| ------------------ | ----------------------- |
| `__shfl_sync`      | 从指定 lane 获取值      |
| `__shfl_up_sync`   | 从低 lane 获取值        |
| `__shfl_down_sync` | 从高 lane 获取值        |
| `__shfl_xor_sync`  | 与 XOR 偏移的 lane 交换 |

**优势**：

1. 不需要共享内存
2. 延迟比共享内存低
3. 无 Bank 冲突

## 多级归约

### 单 Block 不够

如果数据量超过单 Block 能处理的范围，需要多级归约：

```
Level 1: 每个 Block 归约自己的部分 → gridDim 个部分和
Level 2: 再启动一个 Kernel 归约这些部分和
...
重复直到只剩一个值
```

### 实现

```cuda
void reduce_multi_level(float *d_data, float *d_result, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);
    
    float *d_partial;
    cudaMalloc(&d_partial, numBlocks * sizeof(float));
    
    // Level 1
    reduce_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>
        (d_data, d_partial, n);
    
    // 后续 Level
    while (numBlocks > 1) {
        int n_next = numBlocks;
        numBlocks = (n_next + blockSize * 2 - 1) / (blockSize * 2);
        
        reduce_kernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>
            (d_partial, d_partial, n_next);
    }
    
    // 拷贝最终结果
    cudaMemcpy(d_result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);
}
```

### 优化：原子累加

如果只需要最终和，可以用原子操作避免多级：

```cuda
__global__ void reduce_atomic(float *g_data, float *g_result, int n) {
    // ... 常规 Block 内归约 ...
    
    if (tid == 0) {
        atomicAdd(g_result, sdata[0]);  // 直接累加到结果
    }
}
```

**注意**：需要先将 `g_result` 初始化为 0。

## 分支发散深入分析

### 发散的代价

当 Warp 内线程走不同分支时：

```cuda
if (condition) {
    doA();  // 部分线程
} else {
    doB();  // 其他线程
}
```

硬件执行：

1. 所有线程执行 doA()，不满足条件的线程结果被丢弃
2. 所有线程执行 doB()，满足条件的线程结果被丢弃
3. 总时间 = doA() + doB()

**发散程度**影响性能：

| 发散比例     | 性能影响    |
| ------------ | ----------- |
| 0/32         | 无影响      |
| 1/32         | 几乎无影响  |
| 16/32        | 约 50% 性能 |
| 按 Warp 边界 | 无发散      |

### 最小化发散的策略

**1. 让条件按 Warp 对齐**

```cuda
// 差：混合发散
if (tid % 2 == 0) { ... }

// 好：Warp 内无发散
if (tid < 128) { ... }  // 前 4 个 Warp vs 后 4 个 Warp
```

**2. 用算术替代分支**

```cuda
// 有分支
if (a > b) result = a;
else result = b;

// 无分支
result = (a > b) * a + (a <= b) * b;

// 更好：用内置函数
result = max(a, b);
```

**3. 预先分离数据**

```cuda
// 差：运行时分支
if (data[i] > threshold) processA();
else processB();

// 好：预处理分离
// Kernel 1: 分离数据
// Kernel 2: 批量处理 A 类
// Kernel 3: 批量处理 B 类
```

### 归约中的发散控制

朴素 vs 交错的对比：

```
朴素 (stride=1):
  Warp 0: threads 0,2,4,...,30 活跃（16个）
  ↪ 每个 Warp 都有 50% 发散

交错 (stride=128):
  Warp 0-3: 全部活跃
  Warp 4-7: 全部空闲
  ↪ 没有 Warp 内发散！
```

## 性能对比

以 2²⁴（约1600万）个单精度浮点数求和为例：

| 优化版本          | 带宽利用率 | 相对性能 |
| ----------------- | ---------- | -------- |
| 朴素相邻配对      | 4%         | 1×       |
| 交错配对          | 8%         | 2×       |
| 首次加载归约      | 16%        | 4×       |
| 展开最后线程束    | 25%        | 6×       |
| 完全展开          | 40%        | 10×      |
| + 线程束 Shuffle  | 60%        | 15×      |

**测试环境**：
- GPU：NVIDIA RTX 3090（10496 CUDA 核心）
- 数据量：16,777,216 个 float（64MB）
- 块大小：256线程

带宽利用率达到60%已经很高——剩余开销来自核函数启动、同步等不可避免的开销。

## 其他归约操作

### 最大值/最小值

```cuda
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}
```

### 带索引的最大值

返回最大值及其位置：

```cuda
__device__ void warpReduceArgMax(float &val, int &idx) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}
```

### 点积

两个向量的点积 = 逐元素乘法 + 求和归约：

```cuda
__global__ void dotProduct(float *a, float *b, float *result, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0;
    while (i < n) {
        sum += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // 归约
    // ...
}
```

## CUB 库

### 为什么用库

手写归约容易出错，且难以覆盖所有优化。NVIDIA 的 CUB 库提供高度优化的实现：

```cuda
#include <cub/cub.cuh>

void reduceWithCub(float *d_data, float *d_result, int n) {
    // 确定临时存储大小
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, d_result, n);
    
    // 分配临时存储
    cudaMalloc(&d_temp, temp_bytes);
    
    // 执行归约
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_data, d_result, n);
    
    cudaFree(d_temp);
}
```

### CUB 提供的归约

| 函数                   | 功能         |
| ---------------------- | ------------ |
| `DeviceReduce::Sum`    | 求和         |
| `DeviceReduce::Max`    | 最大值       |
| `DeviceReduce::Min`    | 最小值       |
| `DeviceReduce::ArgMax` | 最大值及索引 |
| `DeviceReduce::ArgMin` | 最小值及索引 |
| `DeviceReduce::Reduce` | 自定义操作符 |

### Block 级归约

```cuda
#include <cub/cub.cuh>

__global__ void myKernel(float *data, float *result) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    float val = data[threadIdx.x];
    float sum = BlockReduce(temp_storage).Sum(val);
    
    if (threadIdx.x == 0) {
        result[blockIdx.x] = sum;
    }
}
```

## 小结

第十章系统讲解了并行归约：

**树形归约**：log N 步完成，但朴素实现有严重的分支发散和 Bank 冲突。

**交错配对**：让活跃线程连续，消除 Warp 内发散。这是最关键的优化。

**首次加载归约**：Grid-stride loop 让每线程处理多个元素，减少 Block 数和同步开销。

**展开优化**：利用 Warp 内隐式同步，消除最后几层的 `__syncthreads()`。完全展开进一步消除循环开销。

**Warp Shuffle**：现代 GPU 的利器，直接交换寄存器，无需共享内存。

**分支发散**：按 Warp 边界划分条件，用算术替代分支，是通用的发散最小化技术。

**CUB 库**：生产环境优先使用库，它集成了所有优化且经过充分测试。

归约是并行计算的基础模式，卷积的边界处理、直方图的最终合并、神经网络的 Softmax 都用到。下一章学习前缀和——另一个基础且强大的并行原语。

---

## 🚀 下一步

---

## 📚 参考资料

- PMPP 第四版 Chapter 10
- [第十章：归约和最小化发散](https://smarter.xin/posts/43b40d12/)

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
