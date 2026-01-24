---
title: PMPP-第六章：性能方面的考虑
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 性能优化
  - 内存合并
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: 220818c3
date: 2026-01-16 17:11:58
---

## 前言

第五章学习了内存层次和分块（Tiling）技术，矩阵乘法性能提升了10倍。但这只是开始——实际优化中还有很多细节会影响性能。第六章系统梳理这些性能因素：内存合并、分支发散、资源分配、指令吞吐量等。本章内容偏重工程实践，掌握这些技巧，可以让代码性能进一步提升数倍。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 内存合并（Memory Coalescing）

### 全局内存的硬件特性

全局内存（DRAM）不是按单个字节访问的，而是按**事务（Transaction）**批量访问：

- 每次事务读取一个**内存段（Memory Segment）**
- 段大小通常是 32 字节或 128 字节
- 段必须对齐（起始地址是段大小的倍数）

这意味着：读 1 个 float（4 字节）和读 32 个 float（128 字节），如果落在同一个段，硬件开销相同。

### 合并访问的条件

当 Warp 中的 32 个线程访问**连续且对齐**的内存地址时，这些访问可以合并成最少的事务：

**理想情况**：32 线程访问连续 128 字节

```cuda
// 合并访问（1 次 128B 事务）
float val = data[threadIdx.x];
// 线程 0 访问 data[0], 线程 1 访问 data[1], ...
```

**糟糕情况**：32 线程访问跨步地址

```cuda
// 非合并访问（可能 32 次事务！）
float val = data[threadIdx.x * 32];
// 线程 0 访问 data[0], 线程 1 访问 data[32], ...
// 每个访问落在不同段
```

### 量化影响

假设每次事务延迟相同，合并访问带来的加速：

| 访问模式 | 事务数 | 相对带宽 |
| -------- | ------ | -------- |
| 完全合并 | 1      | 100%     |
| 步长 2   | 2      | 50%      |
| 步长 4   | 4      | 25%      |
| 步长 32  | 32     | 3%       |

步长越大，带宽利用率越低。

### 矩阵访问模式的影响

考虑行主序矩阵 `M[row][col]`，一维索引为 `M[row * width + col]`：

**按行遍历（合并访问）**：

```cuda
// 同一线程束的线程，row 相同，col 连续
int idx = row * width + threadIdx.x;
float val = M[idx];  // 连续地址，实现合并访问
```

**按列遍历（非合并访问）**：

```cuda
// 同一线程束的线程，col 相同，row 连续
int idx = threadIdx.x * width + col;
float val = M[idx];  // 步长 = width，无法合并
```

对于 width = 1024 的矩阵，步长为 1024×4 = 4096 字节，远超内存段大小，每个线程访问不同的内存段，无法合并。

### 矩阵转置优化

矩阵转置 `B[j][i] = A[i][j]`，必然有一个矩阵按列访问。解决方案：**用共享内存做中转**。

```cuda
__global__ void transposeCoalesced(float *A, float *B, int width) {
    __shared__ float tile[32][33];  // 注意 padding 防 bank 冲突
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // 合并读取 A（按行）
    if (x < width && y < width)
        tile[threadIdx.y][threadIdx.x] = A[y * width + x];
    
    __syncthreads();
    
    // 交换 block 索引
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // 合并写入 B（按行，但数据来自 tile 的列）
    if (x < width && y < width)
        B[y * width + x] = tile[threadIdx.x][threadIdx.y];
}
```

**关键**：共享内存没有合并要求，可以任意顺序访问（只需注意 Bank 冲突）。通过共享内存"转换"访问模式，让全局内存读写都是合并的。

## 分区露营（Partition Camping）

### DRAM 的分区结构

全局内存由多个**分区（Partition）**组成，每个分区有独立的访问通道。地址到分区的映射通常是：

```
分区 = (地址 / 256) % 分区数
```

如果所有访问都落在同一分区，其他分区空闲，带宽只有 1/N。

### 何时发生

当多个 Warp 访问地址"步调一致"时：

```cuda
// 假设每个 block 处理矩阵的一列
int col = blockIdx.x;
for (int row = 0; row < height; row++) {
    float val = M[row * width + col];  // 所有 block 同步访问
}
```

如果 width 恰好是分区数的倍数，所有 block 同时访问同一分区。

### 解决方案

1. **调整数据布局**：加 padding 打破对齐
2. **调整访问顺序**：让不同 block 访问不同分区
3. **通常问题不大**：现代 GPU 分区多，自动调度能力强

分区露营在早期 GPU 上是严重问题，现代架构（Ampere+）影响较小，但仍值得注意。

## 指令混合与吞吐量

### 不同指令的吞吐量

并非所有操作速度相同：

| 操作类型       | 吞吐量（FLOP/周期/SM） | 相对速度 |
| -------------- | ---------------------- | -------- |
| FP32 加法/乘法 | 128                    | 1×       |
| FP32 FMA       | 128                    | 1×       |
| FP32 除法      | 16                     | 1/8      |
| FP32 特殊函数  | 32                     | 1/4      |
| FP64 加法/乘法 | 64（或更少）           | 1/2      |
| 整数乘法       | 64                     | 1/2      |

**注意**：除法和特殊函数（sin/cos/exp）慢很多。

### 优化策略

**用乘法替代除法**：

```cuda
// 慢
float y = x / 3.0f;

// 快（预计算倒数）
float inv3 = 1.0f / 3.0f;  // 常量，编译时计算
float y = x * inv3;
```

**用近似函数**：

```cuda
// 精确但慢
float y = sinf(x);

// 快速近似（误差略大）
float y = __sinf(x);  // 内置快速版本
```

**减少特殊函数调用**：

```cuda
// 差：3 次特殊函数
float a = expf(x);
float b = expf(y);
float c = expf(z);

// 好：利用数学性质
float abc = expf(x + y + z);
```

### 混合精度

如果精度允许，用 FP16 或 TF32：

```cuda
// 使用 half 精度（需要 include cuda_fp16.h）
half x = __float2half(input);
half y = __hmul(x, x);  // FP16 乘法，吞吐量更高
float result = __half2float(y);
```

Tensor Core 可以做 FP16/TF32 矩阵乘法，吞吐量比 FP32 CUDA 核心高数倍。

## 线程粒度

### 每线程工作量

线程粒度指每个线程处理的数据量。两个极端：

**细粒度**：每线程处理 1 个元素

```cuda
// 每线程 1 个元素
int i = blockIdx.x * blockDim.x + threadIdx.x;
C[i] = A[i] + B[i];
```

**粗粒度**：每线程处理多个元素

```cuda
// 每线程 4 个元素
int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
for (int i = 0; i < 4; i++) {
    C[base + i] = A[base + i] + B[base + i];
}
```

### 粗粒度的优势

1. **减少启动开销**：更少的线程总数
2. **寄存器复用**：循环变量、指针可复用
3. **指令级并行**：循环展开后多条指令可流水
4. **减少同步**：每线程更多独立工作

### 粗粒度的风险

1. **负载不均衡**：如果元素数不是粒度倍数，最后一批线程工作量不同
2. **占用率下降**：每线程更多寄存器，可能降低并行度
3. **复杂边界处理**：需要额外检查

### 实践建议

```cuda
// 每线程处理 4 个元素，带边界检查
__global__ void vectorAddCoarsened(float *A, float *B, float *C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop
    for (int i = tid; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

**Grid-stride loop** 是经典模式：每线程从自己的起始位置开始，按 grid 大小步进，直到处理完所有元素。兼顾了粒度和灵活性。

## 资源平衡

### 占用率 vs 每线程资源

回顾第四章的占用率概念。资源使用增加会降低占用率：

| 每线程寄存器 | 每 SM 线程数 | 占用率 |
| ------------ | ------------ | ------ |
| 32           | 2048         | 100%   |
| 64           | 1024         | 50%    |
| 128          | 512          | 25%    |
| 256          | 256          | 12.5%  |

类似地，每 Block 共享内存越多，能同时运行的 Block 越少。

### 权衡案例

**场景**：Tiled 矩阵乘法

```cuda
// 方案 A：小 Tile
#define TILE 16
__shared__ float As[16][16];  // 2 KB
__shared__ float Bs[16][16];  // 2 KB
// 共享内存少，可以多 Block，高占用率
// 但 Tile 小，算术强度低

// 方案 B：大 Tile
#define TILE 32
__shared__ float As[32][32];  // 8 KB
__shared__ float Bs[32][32];  // 8 KB
// 共享内存多，Block 少，低占用率
// 但 Tile 大，算术强度高
```

**通常大 Tile 更优**：矩阵乘法是计算密集型，算术强度比占用率更重要。但需要实测验证。

### 使用 Launch Bounds

告诉编译器你的配置，帮助优化寄存器分配：

```cuda
__global__ void __launch_bounds__(256, 4) matMulKernel(...) {
    // 每 Block 256 线程
    // 期望每 SM 至少 4 个 Block
    // 编译器据此限制寄存器使用
}
```

如果不指定，编译器可能过度使用寄存器，导致占用率低于预期。

## Warp 执行效率

### 线程束利用率

如果 Block 大小不是 32 的倍数，最后一个 Warp 有线程空闲：

```cuda
dim3 block(100);  // 100 / 32 = 3.125 → 4 个 warp
// Warp 3 只有 4 个活跃线程，28 个空闲
// warp 利用率 = 100 / 128 = 78%
```

**永远用 32 的倍数作为 Block 大小**。

### 动态分支的额外开销

即使没有分支发散，动态分支本身也有开销：

```cuda
// 有分支开销
if (condition) {
    doWork();
}

// 无分支（条件总为 true）
doWork();
```

编译器无法知道 `condition` 是否总为 true，必须生成分支指令。如果条件可以在编译时确定，用模板或 constexpr。

## 数据预取

### 软件流水

通过提前发起内存请求，与计算重叠：

```cuda
// 无预取
for (int i = 0; i < n; i++) {
    float data = load(i);      // 等待
    process(data);             // 计算
}

// 有预取
float next = load(0);
for (int i = 0; i < n - 1; i++) {
    float curr = next;
    next = load(i + 1);        // 提前请求
    process(curr);             // 同时计算
}
process(next);
```

预取让 load 和 compute 重叠，隐藏部分延迟。

### 在 CUDA 中的应用

```cuda
__global__ void prefetchExample(float *in, float *out, int n) {
    __shared__ float buffer[2][TILE_SIZE];  // 双缓冲
    
    int tile = 0;
    
    // 预加载第一块
    buffer[0][threadIdx.x] = in[threadIdx.x];
    
    for (int i = TILE_SIZE; i < n; i += TILE_SIZE) {
        __syncthreads();
        
        // 加载下一块到另一个缓冲区
        buffer[1 - tile][threadIdx.x] = in[i + threadIdx.x];
        
        // 处理当前块
        out[i - TILE_SIZE + threadIdx.x] = process(buffer[tile][threadIdx.x]);
        
        tile = 1 - tile;  // 切换缓冲区
        __syncthreads();
    }
    
    // 处理最后一块
    out[n - TILE_SIZE + threadIdx.x] = process(buffer[tile][threadIdx.x]);
}
```

双缓冲是经典技术：一个缓冲区处理，另一个加载，交替进行。

## 性能分析工具

### Nsight Compute

NVIDIA 官方 profiler，分析 kernel 性能：

```bash
ncu --set full ./myprogram
```

关键指标：

| 指标                      | 含义           | 目标         |
| ------------------------- | -------------- | ------------ |
| SM Throughput             | 计算单元利用率 | 越高越好     |
| Memory Throughput         | 内存带宽利用率 | 接近硬件峰值 |
| Occupancy                 | 占用率         | 视情况而定   |
| Warp Execution Efficiency | 线程束效率     | 接近 100%    |
| Memory Coalescing         | 合并效率       | 接近 100%    |

### Nsight Systems

分析整体执行流程：

```bash
nsys profile ./myprogram
```

可视化：

- Kernel 执行时间线
- Memory 拷贝时间
- CPU/GPU 交互
- 流和事件

### 常见性能模式

**内存受限**：Memory Throughput 高，SM Throughput 低

- 解决：提高算术强度、用共享内存

**计算受限**：SM Throughput 高，Memory Throughput 低

- 解决：已经很好了，优化算法复杂度

**延迟受限**：两个 Throughput 都低，Occupancy 也低

- 解决：增加并行度，调整 Block 配置

## 优化检查清单

按优先级排序的优化步骤：

### 1. 正确性优先

- [ ] 边界检查完整
- [ ] 同步正确使用
- [ ] 无竞态条件

### 2. 内存优化

- [ ] 全局内存合并访问
- [ ] 使用共享内存减少全局内存访问
- [ ] 避免 Bank 冲突
- [ ] 常量数据用 `__constant__`

### 3. 执行优化

- [ ] Block 大小是 32 的倍数
- [ ] 避免 Warp 内分支发散
- [ ] 减少同步次数
- [ ] 合理线程粒度

### 4. 指令优化

- [ ] 避免除法和特殊函数
- [ ] 用 FMA（编译器自动）
- [ ] 考虑混合精度

### 5. 资源平衡

- [ ] 监控寄存器使用
- [ ] 使用 launch_bounds
- [ ] 实测不同配置

## 小结

第六章是性能优化的工程实践：

**内存合并核心**：Warp 线程访问连续地址，事务数最少。矩阵按行访问好，按列访问差。用共享内存转换访问模式。

**分支发散回顾**：Warp 内走不同分支会串行化。按 Warp 边界划分任务，用算术替代分支。

**指令吞吐量**：除法慢 8 倍，特殊函数慢 4 倍。预计算倒数，用快速近似，考虑混合精度。

**线程粒度**：每线程处理多个元素可提高效率。Grid-stride loop 是通用模式。

**资源平衡**：占用率不是越高越好。计算密集型可接受低占用率，内存密集型需要高占用率来隐藏延迟。

**工具驱动优化**：用 Nsight Compute 分析瓶颈，不要猜测。Memory Throughput 和 SM Throughput 指明优化方向。

这些技巧组合使用，能让代码性能再提升数倍。下一章开始学习具体的并行模式——卷积、规约、前缀和，这些都需要本章的优化技术。

---

## 🚀 下一步

---

## 📚 参考资料

- PMPP 第四版 Chapter 06
- [第六章：性能方面的考虑](https://smarter.xin/posts/220818c3/)

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
