---
title: PMPP-第七章：卷积
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 卷积
  - 常量内存
  - Tiling
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: 1c778456
date: 2026-01-16 21:58:04
---

## 前言

第六章讨论了性能优化的各个方面，现在开始学习具体的并行计算模式。第七章的主角是**卷积（Convolution）**——信号处理和深度学习的核心算子。卷积看似简单，但要高效实现却涉及多个内存层次的配合：常量内存存储卷积核、共享内存实现分块（Tiling）、全局内存处理输入输出。本章将展示如何组合这些技术，实现工业级的卷积核函数。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 卷积基础

### 什么是卷积

卷积是一种数学运算，使用一个小矩阵（**卷积核/滤波器，Kernel/Filter**）在输入数据上滑动，在每个位置计算加权和。

**1D 卷积**：

```
输入：  [1, 2, 3, 4, 5, 6, 7]
卷积核：[1, 2, 1]
输出：  [_, 8, 12, 16, 20, 24, _]

计算 Output[3]：
= Input[2]*Kernel[0] + Input[3]*Kernel[1] + Input[4]*Kernel[2]
= 3*1 + 4*2 + 5*1 = 12
```

**2D 卷积**（图像处理常用）：

```
输入图像                    卷积核 (3×3)
┌───────────────┐           ┌───────┐
│ 1 2 3 4 5     │           │ 1 0 1 │
│ 5 6 7 8 9     │     *     │ 0 1 0 │
│ 9 0 1 2 3     │           │ 1 0 1 │
│ 4 5 6 7 8     │           └───────┘
└───────────────┘
```

### 卷积的应用

卷积无处不在：

| 领域     | 应用                 | 示例                 |
| -------- | -------------------- | -------------------- |
| 图像处理 | 边缘检测、模糊、锐化 | Sobel、Gaussian Blur |
| 音频处理 | 回声、均衡器、降噪   | FIR 滤波器           |
| 深度学习 | 特征提取             | CNN 的核心操作       |
| 物理仿真 | 扩散方程、有限差分   | 热传导、波动方程     |

### 卷积的数学定义

**1D 离散卷积**：

$$
Output[i] = \sum_{j=-r}^{r} Input[i+j] \times Kernel[j+r]
$$

其中 r 是卷积核的"半径"，卷积核大小为 2r+1。

**2D 离散卷积**：

$$
Output[y][x] = \sum_{dy=-r}^{r} \sum_{dx=-r}^{r} Input[y+dy][x+dx] \times Kernel[dy+r][dx+r]
$$

### 边界处理

当卷积窗口超出输入边界时，有几种处理策略：

**1. 零填充（Zero Padding）**：边界外的元素视为 0

```
Padding 前：[1, 2, 3, 4, 5]
Padding 后：[0, 0, 1, 2, 3, 4, 5, 0, 0]
```

**2. 复制填充（Replicate Padding）**：用边界值填充

```
Padding 后：[1, 1, 1, 2, 3, 4, 5, 5, 5]
```

**3. 镜像填充（Reflect Padding）**：镜像边界元素

```
Padding 后：[3, 2, 1, 2, 3, 4, 5, 4, 3]
```

在 CUDA 实现中，零填充最简单——只需在边界判断时返回 0。

## 朴素 1D 卷积实现

### 基本思路

每个线程负责计算一个输出元素：

```cuda
__global__ void conv1d_basic(float *input, float *kernel, float *output,
                              int n, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = kernel_size / 2;  // 卷积核半径
    
    if (i < n) {
        float sum = 0.0f;
        for (int j = -r; j <= r; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < n) {
                sum += input[idx] * kernel[j + r];
            }
            // 边界外隐式为 0（零填充）
        }
        output[i] = sum;
    }
}
```

### 性能问题

这个朴素实现有两个瓶颈：

**1. 卷积核重复读取**

每个线程都要读取整个卷积核。如果有 10 万个线程，卷积核就被读取 10 万次。但卷积核对所有线程是相同的！

**2. 输入数据重复读取**

相邻线程的卷积窗口高度重叠：

```
线程 i   读取：input[i-1], input[i], input[i+1]
线程 i+1 读取：input[i], input[i+1], input[i+2]
                ↑ 共享元素
```

卷积核大小为 2r+1 时，每个输入元素平均被读取 2r+1 次。

## 常量内存优化

### 常量内存特性

第五章提过常量内存，这里深入讲解：

**硬件结构**：

```
┌──────────────────────────────────────┐
│           常量内存 (64 KB)            │
│           Device DRAM                │
└──────────────┬───────────────────────┘
               │ 广播
┌──────────────┴───────────────────────┐
│        常量缓存 (每 SM ~8 KB)         │
│           On-chip Cache              │
└──────────────┬───────────────────────┘
               │ 极低延迟
               ↓
         Warp 中的线程
```

**特点**：

1. **只读**：Kernel 执行期间不能修改
2. **缓存优化**：有专用缓存，命中时延迟极低
3. **广播高效**：同一 Warp 读相同地址时，只需一次访问

**为什么适合卷积核**：

- 卷积核对所有线程都是相同的常量
- 每次卷积操作，所有线程都读相同的 kernel[j]
- 完美匹配常量内存的"广播"特性

### 使用常量内存

**声明**（全局作用域）：

```cuda
#define MAX_KERNEL_SIZE 1025
__constant__ float d_kernel[MAX_KERNEL_SIZE];
```

**Host 端初始化**：

```cuda
float h_kernel[kernel_size];
// ... 填充卷积核数据 ...

cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size * sizeof(float));
```

**Kernel 中使用**：

```cuda
__global__ void conv1d_const(float *input, float *output, 
                              int n, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = kernel_size / 2;
    
    if (i < n) {
        float sum = 0.0f;
        for (int j = -r; j <= r; j++) {
            int idx = i + j;
            if (idx >= 0 && idx < n) {
                sum += input[idx] * d_kernel[j + r];  // 直接使用
            }
        }
        output[i] = sum;
    }
}
```

### 性能提升

常量内存解决了**卷积核重复读取**的问题：

| 版本     | 卷积核读取  | 带宽消耗     |
| -------- | ----------- | ------------ |
| 朴素     | N×K 次      | 高           |
| 常量内存 | K 次 + 缓存 | 极低（广播） |

其中 N 是输出元素数，K 是卷积核大小。

但输入数据的重复读取问题还没解决——这需要 Tiling。

## Tiled 卷积

### 为什么需要 Tiling

回顾输入数据访问模式：

```
Block 内的线程访问：
线程 0:   input[0], input[1], ..., input[k-1]
线程 1:   input[1], input[2], ..., input[k]
...
线程 255: input[255], input[256], ..., input[255+k-1]
```

Block 内的线程访问范围是连续的，有大量重叠。把这部分数据加载到共享内存，就能避免重复的全局内存访问。

### 输入 Tile 设计

关键问题：**一个 Block 的线程需要多大的输入 Tile？**

```
Block 大小：BLOCK_SIZE
卷积核半径：r
卷积核大小：2r + 1

输出范围：[block_start, block_start + BLOCK_SIZE - 1]
输入范围：[block_start - r, block_start + BLOCK_SIZE - 1 + r]
         = block_start - r, block_start + BLOCK_SIZE + r - 1]

输入 Tile 大小 = BLOCK_SIZE + 2r
```

也就是说，要计算 BLOCK_SIZE 个输出，需要读取 BLOCK_SIZE + 2r 个输入元素。

### Halo 元素

输入 Tile 比输出 Tile 大，多出来的部分叫做 **Halo（光晕）元素**：

```
                    Halo                 主体                  Halo
            ┌────────────────┬─────────────────────────┬────────────────┐
输入 Tile:  │ r 个元素       │ BLOCK_SIZE 个元素        │ r 个元素       │
            └────────────────┴─────────────────────────┴────────────────┘
                    ↑                                           ↑
                左 Halo                                      右 Halo
```

Halo 元素是相邻 Block 边界的重叠部分，也叫 **Ghost Cells**。

### 实现策略

有两种加载方式：

**策略 1：统一加载**

所有元素由 BLOCK_SIZE 个线程分工加载，每线程可能加载多个元素：

```cuda
__shared__ float tile[BLOCK_SIZE + 2 * MAX_R];

int tile_size = BLOCK_SIZE + 2 * r;
int loads_per_thread = (tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

for (int i = 0; i < loads_per_thread; i++) {
    int tile_idx = threadIdx.x + i * BLOCK_SIZE;
    if (tile_idx < tile_size) {
        int global_idx = block_start - r + tile_idx;
        tile[tile_idx] = (global_idx >= 0 && global_idx < n) ? 
                          input[global_idx] : 0.0f;
    }
}
```

**策略 2：分区加载（更高效）**

左 Halo、主体、右 Halo 分别由不同线程加载：

```cuda
__shared__ float tile[BLOCK_SIZE + 2 * MAX_R];

int halo_left = r;
int block_start = blockIdx.x * BLOCK_SIZE;

// 左 Halo：前 r 个线程负责
if (threadIdx.x < r) {
    int idx = block_start - r + threadIdx.x;
    tile[threadIdx.x] = (idx >= 0) ? input[idx] : 0.0f;
}

// 主体：所有线程各负责一个
int main_idx = block_start + threadIdx.x;
tile[r + threadIdx.x] = (main_idx < n) ? input[main_idx] : 0.0f;

// 右 Halo：前 r 个线程负责
if (threadIdx.x < r) {
    int idx = block_start + BLOCK_SIZE + threadIdx.x;
    tile[BLOCK_SIZE + r + threadIdx.x] = (idx < n) ? input[idx] : 0.0f;
}

__syncthreads();
```

### 完整的 Tiled 1D 卷积

```cuda
#define BLOCK_SIZE 256
#define MAX_R 512

__constant__ float d_kernel[2 * MAX_R + 1];

__global__ void conv1d_tiled(float *input, float *output, 
                              int n, int r) {
    // 共享内存 Tile
    __shared__ float tile[BLOCK_SIZE + 2 * MAX_R];
    
    int block_start = blockIdx.x * BLOCK_SIZE;
    int global_idx = block_start + threadIdx.x;
    
    // ========== 协作加载 ==========
    
    // 左 Halo
    if (threadIdx.x < r) {
        int idx = block_start - r + threadIdx.x;
        tile[threadIdx.x] = (idx >= 0) ? input[idx] : 0.0f;
    }
    
    // 主体元素
    tile[r + threadIdx.x] = (global_idx < n) ? input[global_idx] : 0.0f;
    
    // 右 Halo
    if (threadIdx.x < r) {
        int idx = block_start + BLOCK_SIZE + threadIdx.x;
        tile[BLOCK_SIZE + r + threadIdx.x] = (idx < n) ? input[idx] : 0.0f;
    }
    
    __syncthreads();
    
    // ========== 计算卷积 ==========
    
    if (global_idx < n) {
        float sum = 0.0f;
        for (int j = 0; j < 2 * r + 1; j++) {
            sum += tile[threadIdx.x + j] * d_kernel[j];
        }
        output[global_idx] = sum;
    }
}
```

### 性能分析

**全局内存访问**：

| 版本  | 每元素读取次数    | 总读取量       |
| ----- | ----------------- | -------------- |
| 朴素  | 2r + 1            | N × (2r + 1)   |
| Tiled | 1 + 2r/BLOCK_SIZE | N × (1 + 2r/B) |

当 BLOCK_SIZE = 256, r = 5 时：

- 朴素：每元素 11 次
- Tiled：每元素 1.04 次

全局内存访问减少约 **10 倍**！

## 2D 卷积

### 扩展到二维

2D 卷积更常见，思路类似但更复杂：

```
输入 Tile 大小：(BLOCK_Y + 2ry) × (BLOCK_X + 2rx)
Halo 元素在四个边和四个角
```

### 2D Tile 结构

```
    ┌─────────────┬───────────────────────────┬─────────────┐
    │  左上角     │        上 Halo            │  右上角     │
    │  (ry×rx)    │      (ry×BLOCK_X)         │  (ry×rx)    │
    ├─────────────┼───────────────────────────┼─────────────┤
    │             │                           │             │
    │  左 Halo    │       主体                │  右 Halo    │
    │ (BLOCK_Y×rx)│   (BLOCK_Y×BLOCK_X)       │(BLOCK_Y×rx) │
    │             │                           │             │
    ├─────────────┼───────────────────────────┼─────────────┤
    │  左下角     │        下 Halo            │  右下角     │
    │  (ry×rx)    │      (ry×BLOCK_X)         │  (ry×rx)    │
    └─────────────┴───────────────────────────┴─────────────┘
```

### 简化的 2D 实现

```cuda
#define TILE_X 16
#define TILE_Y 16
#define R 1  // 3×3 卷积核的半径

__constant__ float d_kernel2d[(2*R+1) * (2*R+1)];

__global__ void conv2d_tiled(float *input, float *output,
                              int width, int height) {
    // 共享内存
    __shared__ float tile[TILE_Y + 2*R][TILE_X + 2*R];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * TILE_X, by = blockIdx.y * TILE_Y;
    int gx = bx + tx, gy = by + ty;
    
    // ========== 加载主体 ==========
    int tile_x = tx + R, tile_y = ty + R;
    tile[tile_y][tile_x] = (gx < width && gy < height) ? 
                            input[gy * width + gx] : 0.0f;
    
    // ========== 加载 Halo（边界线程负责）==========
    
    // 上边界
    if (ty < R) {
        int src_y = gy - R;
        tile[ty][tile_x] = (gx < width && src_y >= 0) ? 
                            input[src_y * width + gx] : 0.0f;
    }
    // 下边界
    if (ty >= TILE_Y - R) {
        int src_y = gy + R;
        tile[tile_y + R][tile_x] = (gx < width && src_y < height) ? 
                                    input[src_y * width + gx] : 0.0f;
    }
    // 左边界
    if (tx < R) {
        int src_x = gx - R;
        tile[tile_y][tx] = (src_x >= 0 && gy < height) ? 
                            input[gy * width + src_x] : 0.0f;
    }
    // 右边界
    if (tx >= TILE_X - R) {
        int src_x = gx + R;
        tile[tile_y][tile_x + R] = (src_x < width && gy < height) ? 
                                    input[gy * width + src_x] : 0.0f;
    }
    
    // 四个角（角落线程负责）
    if (tx < R && ty < R) {  // 左上
        int sx = gx - R, sy = gy - R;
        tile[ty][tx] = (sx >= 0 && sy >= 0) ? input[sy * width + sx] : 0.0f;
    }
    if (tx >= TILE_X - R && ty < R) {  // 右上
        int sx = gx + R, sy = gy - R;
        tile[ty][tile_x + R] = (sx < width && sy >= 0) ? 
                                input[sy * width + sx] : 0.0f;
    }
    if (tx < R && ty >= TILE_Y - R) {  // 左下
        int sx = gx - R, sy = gy + R;
        tile[tile_y + R][tx] = (sx >= 0 && sy < height) ? 
                                input[sy * width + sx] : 0.0f;
    }
    if (tx >= TILE_X - R && ty >= TILE_Y - R) {  // 右下
        int sx = gx + R, sy = gy + R;
        tile[tile_y + R][tile_x + R] = (sx < width && sy < height) ?
                                        input[sy * width + sx] : 0.0f;
    }
    
    __syncthreads();
    
    // ========== 计算卷积 ==========
    if (gx < width && gy < height) {
        float sum = 0.0f;
        for (int dy = -R; dy <= R; dy++) {
            for (int dx = -R; dx <= R; dx++) {
                sum += tile[ty + R + dy][tx + R + dx] * 
                       d_kernel2d[(dy + R) * (2*R + 1) + (dx + R)];
            }
        }
        output[gy * width + gx] = sum;
    }
}
```

### 启动配置

```cuda
dim3 block(TILE_X, TILE_Y);  // 16×16 = 256 线程
dim3 grid((width + TILE_X - 1) / TILE_X, 
          (height + TILE_Y - 1) / TILE_Y);

conv2d_tiled<<<grid, block>>>(d_input, d_output, width, height);
```

## 缓存机制

### L1/L2 缓存与 Tiling 的关系

前面我们手动用共享内存实现了 Tiling。但 GPU 还有自动的缓存机制：

**L1 缓存**（每 SM）：

- Ampere 架构：128 KB（可配置与共享内存分割）
- 自动缓存全局内存访问
- 对于小卷积核，可能"免费"获得数据复用

**L2 缓存**（全局共享）：

- 几 MB 容量
- 所有 SM 共享
- 跨 Block 的数据访问可能命中

### 何时依赖缓存 vs 显式 Tiling

| 场景               | 推荐方式     | 原因                 |
| ------------------ | ------------ | -------------------- |
| 小卷积核（3×3）    | 可能只用缓存 | L1 够用，代码简单    |
| 大卷积核（11×11+） | 显式 Tiling  | 数据量大，需精确控制 |
| 大图像、小 kernel  | 显式 Tiling  | 最大化内存带宽利用   |
| 调试/原型          | 先用缓存     | 快速验证正确性       |

### 现代 GPU 的优化建议

1. **先写简单版本**，依赖 L1/L2 缓存
2. **用 Nsight Compute 分析**，看内存吞吐是否成为瓶颈
3. **如果受限于内存带宽**，再添加共享内存 Tiling
4. **常量内存始终是好选择**（对于卷积核）

## 卷积优化总结

### 优化手段层次

```
┌─────────────────────────────────────────────────────────────────┐
│ Level 4: 算法级优化                                              │
│   - FFT 卷积（大卷积核）                                          │
│   - Winograd 算法（特定尺寸）                                     │
│   - Im2col + GEMM（cuDNN 方式）                                   │
├─────────────────────────────────────────────────────────────────┤
│ Level 3: 数据复用                                                 │
│   - 共享内存 Tiling                                               │
│   - 常量内存存储卷积核                                            │
│   - 寄存器缓存局部结果                                            │
├─────────────────────────────────────────────────────────────────┤
│ Level 2: 内存访问模式                                             │
│   - 合并访问全局内存                                              │
│   - 避免 Bank 冲突                                                │
│   - 预取和双缓冲                                                  │
├─────────────────────────────────────────────────────────────────┤
│ Level 1: 基础正确实现                                             │
│   - 边界检查                                                      │
│   - 正确的索引计算                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 选择策略

| 卷积核大小  | 推荐方法          |
| ----------- | ----------------- |
| 3×3         | Tiled + 常量内存  |
| 5×5 ~ 11×11 | Tiled + 常量内存  |
| 大于 11×11  | 考虑 FFT 或 cuDNN |

### cuDNN：工业级选择

实际生产中，卷积通常用 cuDNN：

```cpp
cudnnConvolutionForward(handle,
    &alpha,
    inputDesc, d_input,
    filterDesc, d_filter,
    convDesc,
    algo,  // 自动选择最优算法
    workspace, workspaceSize,
    &beta,
    outputDesc, d_output);
```

cuDNN 会根据输入尺寸、卷积核大小、GPU 架构，自动选择最优实现（直接卷积、FFT、Winograd、Im2col+GEMM 等）。

但理解底层原理仍然重要——知道 cuDNN 在做什么，才能正确调用和调优。

## 实战：图像锐化

### 锐化卷积核

```
     ┌─────┬─────┬─────┐
     │  0  │ -1  │  0  │
     ├─────┼─────┼─────┤
     │ -1  │  5  │ -1  │
     ├─────┼─────┼─────┤
     │  0  │ -1  │  0  │
     └─────┴─────┴─────┘
```

### 完整示例

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16
#define R 1

__constant__ float d_sharpen_kernel[9] = {
    0, -1,  0,
   -1,  5, -1,
    0, -1,  0
};

__global__ void sharpen_image(unsigned char *input, unsigned char *output,
                               int width, int height) {
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE_SIZE + tx;
    int gy = blockIdx.y * TILE_SIZE + ty;
    
    // 加载（简化版，假设边界内）
    if (gx < width && gy < height) {
        tile[ty + 1][tx + 1] = (float)input[gy * width + gx];
    } else {
        tile[ty + 1][tx + 1] = 0.0f;
    }
    
    // 加载 Halo（简化，仅示意）
    if (tx == 0 && gx > 0) 
        tile[ty + 1][0] = input[gy * width + gx - 1];
    if (tx == TILE_SIZE - 1 && gx < width - 1) 
        tile[ty + 1][TILE_SIZE + 1] = input[gy * width + gx + 1];
    if (ty == 0 && gy > 0) 
        tile[0][tx + 1] = input[(gy - 1) * width + gx];
    if (ty == TILE_SIZE - 1 && gy < height - 1) 
        tile[TILE_SIZE + 1][tx + 1] = input[(gy + 1) * width + gx];
    
    // 角落处理省略...
    
    __syncthreads();
    
    // 卷积计算
    if (gx < width && gy < height) {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                sum += tile[ty + 1 + dy][tx + 1 + dx] * 
                       d_sharpen_kernel[(dy + 1) * 3 + (dx + 1)];
            }
        }
        // 钳制到 [0, 255]
        sum = fminf(fmaxf(sum, 0.0f), 255.0f);
        output[gy * width + gx] = (unsigned char)sum;
    }
}
```

## 1D vs 2D 卷积的可分离性

### 可分离卷积

有些 2D 卷积核可以分解为两个 1D 卷积核的外积：

```
                       ┌───┐   ┌─────────────┐
┌───────────┐          │ 1 │   │             │
│ 1  4  6  4  1 │      │ 4 │ × │ 1 4 6 4 1   │
│ 4 16 24 16 4 │  =    │ 6 │   │             │
│ 6 24 36 24 6 │       │ 4 │   │             │
│ 4 16 24 16 4 │       │ 1 │   │             │
│ 1  4  6  4  1 │      └───┘   └─────────────┘
└───────────┘        垂直 1D     水平 1D
   5×5 高斯
```

### 可分离卷积的优势

原始 2D：每像素 5×5 = 25 次乘法
可分离：每像素 5 + 5 = 10 次乘法

**计算量减少 60%！**

### CUDA 实现思路

```cuda
// 第一步：水平卷积（每行）
conv1d_horizontal<<<gridH, blockH>>>(input, temp, width, height, kernel_h);

// 第二步：垂直卷积（每列）
conv1d_vertical<<<gridV, blockV>>>(temp, output, width, height, kernel_v);
```

常见的高斯模糊、盒状模糊都是可分离的，这是实际优化中的重要技巧。

## 小结

第七章围绕卷积，整合了多项内存优化技术：

**常量内存**：卷积核对所有线程相同，放常量内存是自然选择。广播访问让带宽消耗趋近于零。

**Tiled 共享内存**：输入数据有大量重叠访问。用共享内存缓存 Tile，全局内存访问降低到接近 1 次/元素。

**Halo 处理**：2D Tiling 的关键难点。边界、角落需要细心处理。建议分区加载——不同区域由不同线程负责。

**缓存利用**：现代 GPU 的 L1/L2 缓存能自动覆盖部分数据复用。小卷积核、调试阶段可以依赖缓存，正式优化再加 Tiling。

**可分离卷积**：高斯等可分离核，分解成两次 1D 卷积，计算量直接减半。

卷积是深度学习和图像处理的核心算子，优化卷积的思路——常量内存、Tiling、Halo 处理——适用于很多相似的模板计算模式。下一章会学习另一个重要模式——Stencil（模板计算），思路类似但有自己的特点。

---

## 🚀 下一步

---

## 📚 参考资料

- PMPP 第四版 Chapter 07
- [第七章：卷积](https://smarter.xin/posts/1c778456/)

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
