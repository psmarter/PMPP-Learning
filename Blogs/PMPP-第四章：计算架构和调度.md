---
title: PMPP-第四章：计算架构和调度
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - GPU架构
  - Warp
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: bd5d1d6
date: 2026-01-15 13:08:55
---

## 前言

前三章打下了基础：第一章理解"为什么使用 GPU"，第二章学会"如何编写核函数"，第三章掌握"多维数据处理"。但到目前为止，我们只是在使用 GPU，还未深入理解其内部机制。第四章开始深入 GPU 内部——硬件架构如何影响性能，线程如何被调度执行，代码性能差异的根本原因。理解这些硬件细节，才算真正入门 GPU 编程。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## GPU架构概览

### 流式多处理器（SM）

GPU并不是一个巨大的处理器，而是由多个**流式多处理器（Streaming Multiprocessor, SM）**组成的阵列。每个SM是相对独立的执行单元，可以同时运行多个线程块。

以NVIDIA Ampere架构（如RTX 3080）为例：

| 组件          | 数量/规格        |
| ------------- | ---------------- |
| SM数量        | 68               |
| 每SM CUDA核心 | 128              |
| 每SM最大线程  | 1536             |
| 每SM最大Block | 16               |
| 共享内存      | 每SM 128KB可配置 |
| L2缓存        | 5MB              |

不同架构参数不同，但核心思想相同：**大量小核心并行工作**。

### SM内部结构

每个SM包含：

1. **CUDA核心**：执行整数和单精度浮点运算
2. **Tensor核心**：专用矩阵运算（深度学习加速）
3. **特殊功能单元（SFU）**：执行sin/cos/exp等超越函数
4. **加载/存储单元（LD/ST）**：处理内存访问
5. **Warp调度器**：调度线程执行
6. **寄存器文件**：快速存储，每SM约64KB
7. **共享内存**：block内共享的可编程缓存

关键认识：SM是资源池，多个block共享这些资源。block分配多少资源，决定了SM能同时运行几个block。

### 线程块到SM的映射

启动kernel时，runtime负责将线程块分配到各SM：

```
kernel<<<gridDim, blockDim>>>(args);
         ↓
运行时分配：Block 0→SM0, Block 1→SM2, Block 2→SM1, ...
```

**关键规则**：

- 每个block只在一个SM上执行，不会跨SM
- 一个SM可以同时执行多个block（资源允许的话）
- Block之间没有执行顺序保证
- Block一旦开始执行，会运行到结束

这种设计的好处：block完全独立，硬件可以自由调度，适应不同规模的GPU。

## 线程束（Warp）：执行的基本单位

### 什么是线程束

**线程束（Warp）是32个连续线程组成的执行单位**。这是 NVIDIA GPU 的核心设计，理解线程束就理解了 GPU 执行模型的一半。

```
块（Block）包含256个线程
├── 线程束 0：线程 0-31
├── 线程束 1：线程 32-63
├── 线程束 2：线程 64-95
├── 线程束 3：线程 96-127
├── 线程束 4：线程 128-159
├── 线程束 5：线程 160-191
├── 线程束 6：线程 192-223
└── 线程束 7：线程 224-255
```

**为什么是32个**？硬件设计决定的。每个 SM 有一定数量的计算单元，32是效率最优的分组大小。

### SIMT 执行模型

GPU 采用 **SIMT（Single Instruction Multiple Thread，单指令多线程）** 模型：

- 同一线程束内的32个线程，在同一时钟周期执行相同指令
- 但操作不同的数据（不同的寄存器）
- 类似 SIMD，但更灵活（线程可以有独立状态）

**举例**：

```cuda
C[i] = A[i] + B[i];
```

线程束中的32个线程同时执行"加法"指令：

```
线程束 0 执行：
线程 0：C[0] = A[0] + B[0]  ─┐
线程 1：C[1] = A[1] + B[1]   │ 同一条 add 指令
...                         │ 同一时钟周期
线程 31：C[31] = A[31] + B[31]─┘
```

硬件只需取一次指令，就能完成32个操作。这就是 GPU 高吞吐的来源。

### 线程束调度

每个 SM 有多个线程束调度器（如 Ampere 架构有4个）。每个时钟周期，调度器选择一个就绪的线程束发射指令：

```
周期1：线程束调度器0选择线程束3，发射 add 指令
周期2：线程束调度器0选择线程束7，发射 load 指令
周期3：线程束调度器0选择线程束0，发射 mul 指令
...
```

**关键**：线程束之间是独立调度的。如果线程束3在等待内存数据，调度器可以切换到线程束7执行，避免浪费时钟周期。

## 控制流发散：性能杀手

### 什么是分支发散

当warp内的线程走不同分支时，就发生**控制流发散（Control Divergence）**：

```cuda
if (threadIdx.x < 16) {
    // 分支A
    A[i] = ...;
} else {
    // 分支B
    B[i] = ...;
}
```

问题来了：32个线程必须执行相同指令，但这里线程0-15需要执行分支A，线程16-31需要执行分支B。怎么办？

### 硬件如何处理

GPU采用**谓词执行（Predicated Execution）**：

1. 所有线程先执行分支A，但只有满足条件的线程写入结果
2. 再执行分支B，只有不满足条件的线程写入结果

```
实际执行序列:
Step 1: 执行分支A（线程0-15活跃，16-31抑制）
Step 2: 执行分支B（线程0-15抑制，16-31活跃）
```

**代价**：两个分支都被执行，时间翻倍。如果有n个分支，最坏情况性能降为1/n。

### 性能影响量化

**场景1**：无发散

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];  // 所有线程执行相同操作
```

100%效率，理想情况。

**场景2**：warp内发散

```cuda
if (threadIdx.x % 2 == 0) {
    C[i] = A[i] + B[i];
} else {
    C[i] = A[i] * B[i];
}
```

每个warp一半执行加法，一半执行乘法。效率约50%。

**场景3**：warp间无发散

```cuda
if (threadIdx.x < 32) {  // warp 0整体走这边
    C[i] = A[i] + B[i];
} else {                  // warp 1-7整体走这边
    C[i] = A[i] * B[i];
}
```

每个warp内部不发散，效率接近100%。

**核心原则**：分支发散只在warp内部有开销。让同一warp的线程走相同分支，就不损失性能。

### 避免发散的技巧

**技巧1**：按warp边界划分任务

```cuda
// 差：奇偶分支，warp内发散
if (threadIdx.x % 2 == 0) { taskA(); }
else { taskB(); }

// 好：按warp划分，warp间分工
if (threadIdx.x / 32 == 0) { taskA(); }  // warp 0
else { taskB(); }                         // 其他warp
```

**技巧2**：重组数据

如果必须处理不同类型的元素，可以先排序，让同类元素落在同一warp。

```cuda
// 原始：type分布随机，每个warp都发散
if (type[i] == 0) { processType0(); }
else { processType1(); }

// 优化：按type排序后处理，warp内type相同
sort_by_type(data);  // 预处理
kernel<<<...>>>(sorted_data);
```

**技巧3**：用算术替代分支

```cuda
// 分支版本
if (x > 0) { y = x; }
else { y = -x; }

// 无分支版本（ReLU可以这样）
y = x * (x > 0);  // 利用布尔转int
// 或用内置函数
y = fabs(x);
```

## 延迟隐藏：GPU的核心优化策略

### 什么是延迟

**延迟（Latency）**是指令从发出到完成需要的时钟周期：

| 操作              | 典型延迟    |
| ----------------- | ----------- |
| 算术运算          | 4-8周期     |
| 共享内存          | 20-30周期   |
| 全局内存          | 200-400周期 |
| 特殊函数(sin/exp) | 20-40周期   |

全局内存访问是大头，400周期意味着读一次数据的时间可以做100次计算。

### GPU如何隐藏延迟

CPU用复杂的乱序执行、预取来隐藏延迟。GPU用更简单粗暴的方法：**大量线程+快速切换**。

```
时钟周期推演:
周期1: Warp0发起load A[0]（需要400周期返回）
周期2: Warp1发起load A[32]
周期3: Warp2发起load A[64]
...（继续切换执行其他warp）
周期400: Warp0的数据到了，可以继续执行
周期401: 切回Warp0执行计算
```

只要有足够多的warp可以切换，计算单元就不会空闲。这就是**延迟隐藏**。

### 延迟隐藏的数学

假设：

- 全局内存延迟400周期
- 算术指令吞吐量1个/周期（每个调度器）
- 每SM有4个调度器

完全隐藏延迟需要的最小warp数：

```
需要活跃warp数 = 延迟 / 吞吐量 = 400 / 1 = 400（每调度器）
每SM需要 = 400 / 4 = 100 warp（理论最小值）
```

实际上每SM最多64 warp（2048线程），所以全局内存延迟很难完全隐藏。这就是为什么要尽量用共享内存和寄存器。

## 占用率（Occupancy）

### 定义

**占用率（Occupancy）**是"SM上活跃warp数"与"SM最大支持warp数"的比值：

```
占用率 = 活跃Warp数 / SM最大Warp数
```

例如，Ampere SM最多支持64 warp（2048线程）。如果实际有32 warp在运行：

```
占用率 = 32 / 64 = 50%
```

### 影响占用率的资源

三大限制因素：

**1. 每block线程数**

```cuda
dim3 block(1024);  // 每block 1024线程 = 32 warp
// SM最多16 block，但...
// SM最多2048线程，所以只能放2个这样的block
// 活跃 = 2 × 32 = 64 warp，占用率100%

dim3 block(64);    // 每block 64线程 = 2 warp
// SM最多16 block，可以放16个
// 活跃 = 16 × 2 = 32 warp，占用率50%
```

**2. 寄存器使用**

每SM寄存器数量有限（如65536个）。kernel用的寄存器越多，能同时运行的线程越少：

```
假设kernel用64个寄存器/线程
每SM可支持线程 = 65536 / 64 = 1024线程 = 32 warp
占用率 = 32 / 64 = 50%

假设kernel用32个寄存器/线程
每SM可支持线程 = 65536 / 32 = 2048线程 = 64 warp
占用率 = 100%
```

**3. 共享内存使用**

每SM共享内存有限（如96KB）。block用的共享内存越多，能同时运行的block越少：

```
假设block用48KB共享内存
每SM可运行block = 96 / 48 = 2个
假设每block 256线程 = 8 warp
活跃 = 2 × 8 = 16 warp
占用率 = 16 / 64 = 25%
```

### 计算占用率

使用CUDA Occupancy Calculator或API：

```cuda
#include <cuda_runtime.h>

int blockSize = 256;
int minGridSize, gridSize;

// 自动计算最优block大小
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                    myKernel, 0, 0);

// 计算给定配置的占用率
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, 
                                               myKernel, blockSize, 0);
int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
float occupancy = (float)(numBlocks * blockSize) / prop.maxThreadsPerMultiProcessor;
printf("占用率: %.2f%%\n", occupancy * 100);
```

### 占用率与性能

**高占用率不一定意味着高性能**。这是常见误解。

```
场景1：计算密集型kernel
- 每线程用大量寄存器做复杂计算
- 占用率可能只有25%
- 但计算单元利用率高，性能好

场景2：内存密集型kernel
- 每线程操作简单，主要在读写内存
- 需要高占用率来隐藏延迟
- 50%占用率可能不够
```

**原则**：

- 内存受限kernel：提高占用率有帮助
- 计算受限kernel：占用率够用就行，优先利用好计算资源
- 最佳实践：从分析实际瓶颈入手，而不是盲目追求占用率

## 资源限制汇总

以Ampere架构为例：

| 资源     | 每SM限制         | 每Block限制     |
| -------- | ---------------- | --------------- |
| 线程数   | 2048             | 1024            |
| Block数  | 16               | -               |
| Warp数   | 64               | 32              |
| 寄存器   | 65536            | 65536           |
| 共享内存 | 可配置,最大164KB | 可配置,最大99KB |

### Block大小选择

**实用建议**：

1. **128-256是安全起点**：满足warp整数倍，不太大不太小
2. **必须是32的倍数**：否则最后一个warp浪费线程
3. **考虑共享内存需求**：block越大，共享内存分摊效率越高
4. **实测为王**：最终还是要profile不同配置的性能

```cuda
// 不好：不是32的倍数
dim3 block(100);  // 最后warp只有4个有效线程

// 好：32的倍数
dim3 block(128);  // 4个完整warp

// 也好
dim3 block(256);  // 8个完整warp
```

## 编译器优化与调试

### 查看寄存器使用

编译时加`--ptxas-options=-v`：

```bash
nvcc -O3 --ptxas-options=-v mykernel.cu -o mykernel
```

输出：

```
ptxas info : Compiling entry function 'mykernel'
ptxas info : Function properties for mykernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 32 registers, 336 bytes cmem[0]
```

32个寄存器/线程，没有溢出到local memory。

### 控制寄存器数量

```cuda
__global__ void __launch_bounds__(256, 4) myKernel(...) {
    // 提示编译器：每block 256线程，每SM至少4 block
    // 编译器会据此优化寄存器分配
}
```

或编译选项：

```bash
nvcc -maxrregcount=32 mykernel.cu  # 限制每线程最多32寄存器
```

**权衡**：限制寄存器可能导致溢出到local memory（很慢），需要平衡。

### Nsight Compute分析

使用NVIDIA的profiler分析实际瓶颈：

```bash
ncu --set full ./mykernel
```

关注：

- Warp执行效率（分支发散）
- 内存吞吐率
- 计算吞吐率
- 占用率

这比猜测有效得多。

## 实例分析：矩阵乘法的Block配置

回顾第三章的矩阵乘法：

```cuda
__global__ void matMul(float *M, float *N, float *P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = sum;
    }
}
```

**配置选择**：

| Block大小  | Warp数 | 分析                       |
| ---------- | ------ | -------------------------- |
| 8×8=64     | 2      | 太小，warp少，延迟隐藏不足 |
| 16×16=256  | 8      | 合适，每SM可放多个block    |
| 32×32=1024 | 32     | 接近限制，灵活性差         |

**16×16通常是二维问题的好选择**。

**进一步优化**（第5章会详细讲）：

- 用共享内存做tiling
- 减少全局内存访问
- 算术强度从0.25提升到10+

## 小结

第四章揭示了GPU的内部机制：

**架构认识**：SM是独立执行单元，多个block分配到SM共享资源。理解资源限制（寄存器、共享内存、线程数），就理解了为什么某些配置性能差。

**Warp本质**：32线程一组，SIMT执行。所有优化都围绕warp展开——让warp内线程做相同的事（避免发散），让足够多的warp参与执行（隐藏延迟）。

**发散代价**：控制流发散是性能杀手。按warp边界划分任务、用算术替代分支、重组数据，这些技巧很实用。

**延迟与占用率**：高占用率帮助隐藏延迟，但不是越高越好。计算密集型kernel可能不需要那么高占用率。关键是分析实际瓶颈。

**调优思路**：

- Block大小选128-256，32的倍数
- 检查寄存器使用，控制溢出
- 用profiler分析，不要猜

理解了这些硬件知识，第五章的共享内存（Shared Memory）优化就容易理解了。分块（Tiling）的本质就是利用共享内存减少全局内存访问，提高算术强度，充分利用计算单元。

---

## 🚀 下一步

---

## 📚 参考资料

- PMPP 第四版 Chapter 04
- [第四章：计算架构和调度](https://smarter.xin/posts/bd5d1d6/)

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
