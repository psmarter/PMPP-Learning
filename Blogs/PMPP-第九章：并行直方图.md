---
title: PMPP-第九章：并行直方图
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 直方图
  - 原子操作
  - 私有化
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: d29973f1
date: 2026-01-17 20:54:04
---

## 前言

前几章学的卷积、模板都是"规则"的并行模式——输出位置固定，每个线程知道自己写哪里。但很多实际问题不是这样的，比如**直方图**：每个输入元素决定更新哪个输出桶，多个线程可能同时更新同一个桶。这就是**输出冲突**问题。第九章讲解如何用**原子操作**和**私有化**技术解决这类问题。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 直方图基础

### 什么是直方图

直方图统计数据的分布。给定一组数据，计算每个值（或区间）出现的次数。

**例子**：统计文本中每个字母的出现次数

```
输入: "hello world"
输出: h:1, e:1, l:3, o:2, w:1, r:1, d:1, 空格:1
```

**图像直方图**：统计每个灰度值（0-255）的像素数量

```
输入: 256×256 图像
输出: histogram[256]，每个元素是该灰度值的像素计数
```

### 串行实现

```c
void histogram_sequential(unsigned char *data, int *histogram, int n) {
    // 初始化
    for (int i = 0; i < 256; i++) {
        histogram[i] = 0;
    }
    // 统计
    for (int i = 0; i < n; i++) {
        histogram[data[i]]++;
    }
}
```

时间复杂度 O(n)，空间复杂度 O(桶数)。

### 并行化的挑战

尝试直接并行化：

```cuda
__global__ void histogram_naive(unsigned char *data, int *histogram, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        histogram[data[i]]++;  // 危险！读-改-写竞争
    }
}
```

**问题**：多个线程可能同时读取同一个 `histogram[k]`，各自加 1，然后写回。结果只加了 1 次而不是多次。

```
线程 A: 读 histogram[5] = 10
线程 B: 读 histogram[5] = 10
线程 A: 写 histogram[5] = 11
线程 B: 写 histogram[5] = 11  // 应该是 12！
```

这就是**竞态条件（Race Condition）**。

## 原子操作

### 什么是原子操作

**原子操作**：不可分割的操作。整个"读-改-写"过程要么全部完成，要么完全不执行，不会被其他线程打断。

CUDA 提供的原子函数：

| 函数               | 操作             | 返回值 |
| ------------------ | ---------------- | ------ |
| `atomicAdd`        | `*addr += val`   | 旧值   |
| `atomicSub`        | `*addr -= val`   | 旧值   |
| `atomicMax`        | `*addr = max()`  | 旧值   |
| `atomicMin`        | `*addr = min()`  | 旧值   |
| `atomicExch`       | `*addr = val`    | 旧值   |
| `atomicCAS`        | compare-and-swap | 旧值   |
| `atomicAnd/Or/Xor` | 位操作           | 旧值   |

### 使用原子操作的直方图

```cuda
__global__ void histogram_atomic(unsigned char *data, int *histogram, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&histogram[data[i]], 1);  // 原子加
    }
}
```

**正确性保证**：`atomicAdd` 确保每次增量都被正确计入。

### 原子操作的开销

原子操作比普通操作慢得多：

```
普通写入：~4 周期
原子操作：~数百周期（取决于争用程度）
```

**原因**：

1. **串行化**：同一地址的原子操作必须排队执行
2. **缓存一致性**：需要协调多个 SM 的缓存
3. **内存事务**：需要往返全局内存

**争用程度**影响很大：

| 场景       | 桶数 | 争用程度 | 性能     |
| ---------- | ---- | -------- | -------- |
| 字母统计   | 26   | 极高     | 很慢     |
| 灰度直方图 | 256  | 高       | 较慢     |
| 颜色直方图 | 16M  | 低       | 接近峰值 |

桶越多，争用越低，性能越好。

## 私有化（Privatization）

### 核心思想

**私有化**：每个线程/块维护自己的私有直方图，最后合并。

```
原本：所有线程 → 全局直方图（高争用）
私有化：
  线程/块 → 私有直方图（无争用）
  最后：私有直方图 → 全局直方图（一次性合并）
```

### 共享内存私有化

每个 Block 用共享内存维护私有直方图：

```cuda
#define NUM_BINS 256

__global__ void histogram_privatized(unsigned char *data, int *histogram, int n) {
    // 私有直方图（共享内存）
    __shared__ int private_hist[NUM_BINS];
    
    // 初始化私有直方图
    if (threadIdx.x < NUM_BINS) {
        private_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // 统计到私有直方图
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (i < n) {
        atomicAdd(&private_hist[data[i]], 1);  // 共享内存原子操作
        i += stride;
    }
    __syncthreads();
    
    // 合并到全局直方图
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], private_hist[threadIdx.x]);
    }
}
```

### 为什么更快

**共享内存原子操作比全局内存快得多**：

| 操作位置 | 延迟      | 带宽      |
| -------- | --------- | --------- |
| 全局内存 | ~400 周期 | ~500 GB/s |
| 共享内存 | ~20 周期  | ~10 TB/s  |

加速比约 20 倍（理想情况）。

**争用也减少**：

```
原本：所有线程争用同一个全局直方图
私有化：
  - Block 内线程争用私有直方图（共享内存，快）
  - Block 间合并时争用全局直方图（但只有 gridDim 次）
```

### 分阶段分析

```
阶段 1（初始化）：NUM_BINS 次写入共享内存
阶段 2（私有统计）：n/gridDim 次共享内存原子操作
阶段 3（合并）：NUM_BINS 次全局内存原子操作
```

全局原子操作从 n 次降到 NUM_BINS × gridDim 次，大幅减少。

## 线程粗化

### Grid-Stride Loop

之前代码已经用了 grid-stride loop：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

while (i < n) {
    // 处理元素 i
    i += stride;
}
```

**优势**：

1. 每个线程处理多个元素，分摊开销
2. Grid 大小可以固定，不随数据量变化
3. 更好的缓存利用

### 连续访问优化

让每个线程处理连续的一段数据：

```cuda
__global__ void histogram_coarsened(unsigned char *data, int *histogram, int n) {
    __shared__ int private_hist[NUM_BINS];
    
    // 初始化
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        private_hist[i] = 0;
    }
    __syncthreads();
    
    // 每线程处理连续的 COARSEN_FACTOR 个元素
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN_FACTOR;
    
    for (int k = 0; k < COARSEN_FACTOR; k++) {
        int idx = base + k;
        if (idx < n) {
            atomicAdd(&private_hist[data[idx]], 1);
        }
    }
    __syncthreads();
    
    // 合并
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], private_hist[i]);
    }
}
```

**优势**：连续访问利于内存合并。

## 聚合（Aggregation）

### 问题

即使用了共享内存私有化，同一 Warp 内的线程可能频繁争用同一个桶。

**例子**：处理全黑图像（所有像素值都是 0）

```
32 个线程同时 atomicAdd(&private_hist[0], 1)
→ 32 次串行化的原子操作
```

### 解决方案：线程束级聚合

先在线程束内统计每个值出现多少次，再做一次原子操作：

```cuda
__global__ void histogram_aggregated(unsigned char *data, int *histogram, int n) {
    __shared__ int private_hist[NUM_BINS];
    
    // 初始化
    if (threadIdx.x < NUM_BINS) {
        private_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        unsigned char value = data[i];
        
        // Warp 级投票：找出同值线程
        unsigned int mask = __match_any_sync(__activemask(), value);
        
        // 只有组内第一个线程执行原子操作
        if (__ffs(mask) - 1 == (threadIdx.x % 32)) {
            atomicAdd(&private_hist[value], __popc(mask));
        }
    }
    __syncthreads();
    
    // 合并到全局
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], private_hist[threadIdx.x]);
    }
}
```

### 线程束级原语

| 函数               | 功能                                         |
| ------------------ | -------------------------------------------- |
| `__match_any_sync` | 返回值相同的线程掩码                         |
| `__ffs`            | 找第一个置位的位（Find First Set）           |
| `__popc`           | 统计置位的位数（Population Count）           |
| `__activemask`     | 当前活跃线程掩码                             |

**`__match_any_sync` 示例**：

```
线程束内前8个线程的值：[5, 3, 5, 5, 2, 3, 5, 2]
__match_any_sync 返回值：
  线程 0,2,3,6 返回 0b01001101（值为5的线程掩码）
  线程 1,5 返回 0b00100010（值为3的线程掩码）
  线程 4,7 返回 0b10010000（值为2的线程掩码）
```

**效果**：原子操作次数从32次降到3次（等于不同值的数量）。对于数据重复率高的场景，性能提升显著。

## 完整优化版本

### 综合所有优化

```cuda
#define BLOCK_SIZE 256
#define NUM_BINS 256
#define COARSEN_FACTOR 4

__global__ void histogram_optimized(unsigned char *data, int *histogram, int n) {
    // 共享内存私有直方图
    __shared__ int private_hist[NUM_BINS];
    
    // 协作初始化
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        private_hist[i] = 0;
    }
    __syncthreads();
    
    // Grid-stride loop + 粗化
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int base = tid * COARSEN_FACTOR; base < n; base += stride * COARSEN_FACTOR) {
        // 加载连续的 COARSEN_FACTOR 个元素
        unsigned char values[COARSEN_FACTOR];
        
        #pragma unroll
        for (int k = 0; k < COARSEN_FACTOR; k++) {
            int idx = base + k;
            values[k] = (idx < n) ? data[idx] : 0xFF;  // 0xFF 作为无效标记
        }
        
        // 逐个处理，使用线程束聚合
        #pragma unroll
        for (int k = 0; k < COARSEN_FACTOR; k++) {
            if (values[k] != 0xFF) {
                unsigned int mask = __match_any_sync(__activemask(), values[k]);
                if (__ffs(mask) - 1 == (threadIdx.x % 32)) {
                    atomicAdd(&private_hist[values[k]], __popc(mask));
                }
            }
        }
    }
    __syncthreads();
    
    // 合并到全局直方图
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (private_hist[i] > 0) {
            atomicAdd(&histogram[i], private_hist[i]);
        }
    }
}
```

### 启动配置

```cuda
int numBlocks = (n + BLOCK_SIZE * COARSEN_FACTOR - 1) / (BLOCK_SIZE * COARSEN_FACTOR);
numBlocks = min(numBlocks, 256);  // 限制 block 数量

histogram_optimized<<<numBlocks, BLOCK_SIZE>>>(d_data, d_histogram, n);
```

### 性能对比

以 1920×1080 灰度图像直方图计算为例：

| 版本              | 相对性能 | 主要瓶颈     |
| ----------------- | -------- | ------------ |
| 朴素全局原子操作  | 1×       | 全局内存争用 |
| 共享内存私有化    | 10×      | 共享内存原子 |
| + 线程粗化        | 15×      | 原子操作     |
| + 线程束聚合      | 25×      | 接近带宽上限 |

**测试环境**：
- 图像大小：1920×1080（约207万像素）
- GPU：NVIDIA RTX 3080（8704 CUDA 核心）
- 桶数：256（灰度值0-255）
- 块大小：256线程

## 其他私有化策略

### 线程级私有化

如果桶数很少（如 8 个），可以用寄存器：

```cuda
__global__ void histogram_register(unsigned char *data, int *histogram, int n) {
    // 每线程私有直方图（寄存器）
    int local_hist[8] = {0};
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (i < n) {
        int bin = data[i] % 8;  // 假设只有 8 个桶
        local_hist[bin]++;
        i += stride;
    }
    
    // 合并到全局
    for (int b = 0; b < 8; b++) {
        atomicAdd(&histogram[b], local_hist[b]);
    }
}
```

**优势**：寄存器最快，无争用。

**限制**：桶数必须很少（寄存器数量有限）。

### 多级私有化

对于大桶数：

```
寄存器（极少桶）→ 共享内存（中等桶）→ 全局内存（大桶数）
```

每级容量递增，速度递减。

## 原子操作的硬件支持

### 支持的数据类型

| 类型     | 原子操作支持 | 备注             |
| -------- | ------------ | ---------------- |
| int      | 全部         | 最常用           |
| unsigned | 全部         |                  |
| float    | atomicAdd    | Kepler+ (CC 3.0) |
| double   | atomicAdd    | Pascal+ (CC 6.0) |
| half     | atomicAdd    | Volta+ (CC 7.0)  |

### 共享内存 vs 全局内存原子

| 特性     | 共享内存原子 | 全局内存原子 |
| -------- | ------------ | ------------ |
| 延迟     | ~20 周期     | ~400 周期    |
| 带宽     | 高           | 低           |
| 争用范围 | Block 内     | 全设备       |
| 适用场景 | 中间结果     | 最终结果     |

### 原子操作实现原理

**Compare-And-Swap (CAS)**：

```cuda
// atomicAdd 的本质实现
__device__ int atomicAdd_manual(int *addr, int val) {
    int old = *addr, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr, assumed, assumed + val);
    } while (old != assumed);
    return old;
}
```

循环直到成功——高争用时可能循环很多次。

## 应用扩展

### 多通道直方图

RGB 图像的三通道直方图：

```cuda
__global__ void histogram_rgb(unsigned char *image, int *hist_r, int *hist_g, int *hist_b, int n) {
    __shared__ int priv_r[256], priv_g[256], priv_b[256];
    
    // 初始化
    // ...
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&priv_r[image[3*i + 0]], 1);
        atomicAdd(&priv_g[image[3*i + 1]], 1);
        atomicAdd(&priv_b[image[3*i + 2]], 1);
    }
    __syncthreads();
    
    // 合并
    // ...
}
```

### 加权直方图

每个数据点有权重：

```cuda
atomicAdd(&histogram[data[i]], weight[i]);
```

用于直方图均衡化等场景。

### 二维直方图

统计两个变量的联合分布：

```cuda
int bin_x = data_x[i] / bin_width_x;
int bin_y = data_y[i] / bin_width_y;
int bin = bin_y * num_bins_x + bin_x;
atomicAdd(&histogram_2d[bin], 1);
```

## 性能调优建议

### 选择策略

| 桶数      | 推荐策略               |
| --------- | ---------------------- |
| < 16      | 寄存器私有化           |
| 16 - 1024 | 共享内存私有化 + 聚合  |
| > 1024    | 直接全局原子（争用低） |

### 关键参数

**Block 大小**：256 或 512，保证足够的并行度。

**粗化因子**：4-16，平衡寄存器压力和计算粒度。

**Grid 大小**：不要太大，否则合并阶段开销增加。

### Nsight 指标

| 指标                    | 含义         | 目标     |
| ----------------------- | ------------ | -------- |
| Atomic Operations       | 原子操作数   | 越少越好 |
| Shared Memory Bandwidth | 共享内存带宽 | 接近峰值 |
| Warp Efficiency         | Warp 利用率  | > 90%    |

## 小结

第九章解决了"输出冲突"问题：

**原子操作**：保证"读-改-写"的原子性，解决竞态条件。但全局内存原子操作很慢，尤其在高争用时。

**私有化**：每个 Block 用共享内存维护私有副本，最后合并。共享内存原子比全局快 20 倍，争用也被限制在 Block 内。

**线程粗化**：每线程处理多个元素，分摊初始化和合并开销。Grid-stride loop 是通用模式。

**线程束聚合**：使用 `__match_any_sync` 找同值线程，只做一次原子操作。在数据重复率高时效果显著。

**策略选择**：桶数决定策略——少桶用寄存器，中桶用共享内存，多桶直接全局原子。

直方图是"归约到多个目标"的典型代表。原子操作和私有化技术也适用于其他类似问题：散射（scatter）、分组聚合、哈希表构建等。下一章将学习另一个重要模式——归约。

---

## 🚀 下一步

---

## 📚 参考资料

- PMPP 第四版 Chapter 09
- [第九章：并行直方图](https://smarter.xin/posts/d29973f1/)

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
