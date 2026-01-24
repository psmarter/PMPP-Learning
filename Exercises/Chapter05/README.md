# 第五章：内存架构和数据局部性

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章深入介绍 GPU 内存优化技术：

- GPU 内存层次结构（寄存器、共享内存、L2 缓存、全局内存）
- 共享内存（Shared Memory）的使用方法
- Tiling 分块技术
- `__syncthreads()` 同步机制
- 内存合并访问（Coalesced Access）
- Bank 冲突和避免方法
- 算术强度和内存带宽分析

**相关博客笔记**：[第五章：内存架构和数据局部性](https://smarter.xin/posts/3bb3179b/)

---

## 💻 代码实现

### Exercise01 - Tiled 矩阵乘法性能对比

本章核心内容是 Tiling 技术，代码对比朴素矩阵乘法与 Tiled 版本的性能差异。

**代码位置**：`Exercise01/`

**功能**：

- **朴素版本**：每个线程直接从全局内存读取整行和整列
- **Tiled 版本**：使用 32×32 共享内存分块，减少全局内存访问

**核心优化**：

```cuda
// Tiled 矩阵乘法核心
__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

for (int ph = 0; ph < numTiles; ++ph) {
    // 协作加载 Tile 到共享内存
    Mds[ty][tx] = M[row * n + ph * TILE_WIDTH + tx];
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * o + col];
    __syncthreads();  // 等待加载完成
    
    // 在共享内存上计算
    for (int k = 0; k < TILE_WIDTH; ++k) {
        Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();  // 等待计算完成
}
```

#### 运行 Exercise01

```bash
cd Exercise01
make
make run
```

#### 预期输出

```text
========================================
  第五章：内存架构和数据局部性
  Tiled Matrix Multiplication Benchmark
========================================

矩阵大小: 1024 × 1024 × 1024
测试迭代次数: 10

=== 正确性验证 ===
✅ 两种方法结果一致！

=== 性能测试 ===
朴素矩阵乘法:    57.273 ms
Tiled 矩阵乘法:  53.636 ms

加速比: 1.07x
```

---

### Exercise02 - 动态 Tile 大小矩阵乘法

本练习演示如何根据硬件规格动态计算最优 Tile 大小，而不是使用硬编码值。

**代码位置**：`Exercise02/`

**功能**：

- **自动计算最优 Tile 宽度**：基于共享内存大小和线程块限制
- **动态共享内存**：运行时分配共享内存大小
- **性能对比**：与固定 Tile 大小版本对比

**核心优化**：

```cuda
// 动态计算最优 Tile 宽度
int calculateOptimalTileWidth(int m, int n, int o) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // 基于共享内存大小计算
    size_t sharedMemPerBlock = prop.sharedMemPerBlock;
    int maxTileWidth = (int)sqrt(sharedMemPerBlock / (2 * sizeof(float)));
    
    // 确保是 Warp 大小的倍数
    maxTileWidth = (maxTileWidth / prop.warpSize) * prop.warpSize;
    
    return min(maxTileWidth, 32);  // 最大 32
}
```

#### 运行 Exercise02

```bash
cd Exercise02
make
make run
```

#### 预期输出

```text
========================================
  第五章：内存架构和数据局部性
  Dynamic Tile Size Matrix Multiplication
========================================

设备: NVIDIA GeForce RTX 4090
计算的最优 Tile 宽度: 32

矩阵大小: 1024 × 1024 × 1024

=== 正确性验证 ===
✅ 结果正确！

=== 性能测试 ===
动态 Tile 矩阵乘法:  52.8 ms
```

---

## 📖 练习题解答

### 练习 1

**题目：** 考虑矩阵加法。能否使用共享内存来减少全局内存带宽消耗？提示：分析每个线程访问的元素，看是否有线程间的共性。

**解答：**

**不能用共享内存优化矩阵加法。**

**分析**：在矩阵加法中，线程之间没有数据复用：

- 线程 (0, 0) 加载 `M[0][0]` 和 `N[0][0]`
- 线程 (124, 12) 加载 `M[124][12]` 和 `N[124][12]`
- 每个元素只被一个线程使用

由于没有跨线程的数据共享，将数据加载到共享内存不会减少全局内存访问次数，反而增加了共享内存写入的开销。

**对比**：矩阵乘法中，`M[i][k]` 被第 i 行的所有线程使用，`N[k][j]` 被第 j 列的所有线程使用，存在大量复用，所以能从共享内存优化中获益。

---

### 练习 2

**题目：** 画出 8×8 矩阵乘法使用 2×2 Tiling 和 4×4 Tiling 的等效图（类似教材图 5.7）。验证全局内存带宽减少确实与 Tile 维度成正比。

**解答：**

**2×2 Tiling**：

```text
┌─────────────────┐    ┌─────────────────┐
│  M (8×8)        │    │  N (8×8)        │
│ ┌──┬──┬──┬──┐   │    │ ┌──┬──┬──┬──┐   │
│ │T1│T2│T3│T4│   │    │ │T1│T2│T3│T4│   │
│ ├──┼──┼──┼──┤   │    │ ├──┼──┼──┼──┤   │
│ │  │  │  │  │   │    │ │  │  │  │  │   │
│ ...             │    │ ...             │
└─────────────────┘    └─────────────────┘

每个 Tile: 2×2 = 4 元素
Tile 数量: (8/2)² = 16 块
全局内存访问: 每元素读 8/2 = 4 次
```

**4×4 Tiling**：

```text
┌─────────────────┐    ┌─────────────────┐
│  M (8×8)        │    │  N (8×8)        │
│ ┌────┬────┐     │    │ ┌────┬────┐     │
│ │ T1 │ T2 │     │    │ │ T1 │ T2 │     │
│ ├────┼────┤     │    │ ├────┼────┤     │
│ │ T3 │ T4 │     │    │ │ T3 │ T4 │     │
│ └────┴────┘     │    │ └────┴────┘     │
└─────────────────┘    └─────────────────┘

每个 Tile: 4×4 = 16 元素
Tile 数量: (8/4)² = 4 块
全局内存访问: 每元素读 8/4 = 2 次
```

**验证**：

| Tile 大小 | 每元素读取次数 | 带宽减少倍数 |
| --------- | -------------- | ------------ |
| 无 Tiling | 8 次           | 1×           |
| 2×2       | 4 次           | 2×           |
| 4×4       | 2 次           | 4×           |

带宽减少与 Tile 维度成正比 ✓

---

### 练习 3

**题目：** 如果忘记使用图 5.9 kernel 中的一个或两个 `__syncthreads()`，会发生什么类型的错误执行行为？

**解答：**

**忘记第一个 `__syncthreads()`（加载后）**：

- 某些线程开始计算时，其他线程可能还未完成加载
- 读取到未初始化或上一轮的旧数据
- 导致计算结果错误

**忘记第二个 `__syncthreads()`（计算后）**：

- 快线程开始加载下一个 Tile，覆盖共享内存
- 慢线程仍在使用被覆盖的数据
- 导致计算结果错误

**两个都忘记**：

- 同时出现上述两种错误
- 加载/计算完全乱序，结果不可预测

---

### 练习 4

**题目：** 假设容量不是寄存器或共享内存的问题，给出一个重要理由说明为什么使用共享内存而不是寄存器来保存从全局内存获取的值是有价值的？

**解答：**

**共享内存可被 Block 内所有线程访问，寄存器只属于单个线程。**

**详细解释**：

- **共享内存**：一个线程加载的值可以被同 Block 的其他线程使用
  - 例如：线程 0 加载 `M[0][0]`，线程 1-31 都可以读取
  - 实现数据复用，减少全局内存访问

- **寄存器**：只有拥有该寄存器的线程可以访问
  - 如果每个线程都需要 `M[0][0]`，必须各自从全局内存加载
  - 无法实现跨线程共享

**应用场景**：

- 矩阵乘法中，M 的一行被多个线程使用 → 适合共享内存
- 每个线程独立的循环变量 → 适合寄存器

---

### 练习 5

**题目：** 对于我们的 Tiled 矩阵乘法 kernel，如果使用 32×32 的 Tile，输入矩阵 M 和 N 的内存带宽使用减少了多少？

**解答：** **减少 32 倍**

**分析**：

**朴素版本**：

- 计算 `P[row][col]` 需要读取 M 的整行（n 个元素）
- P 的每一行有 n 个元素，每个都读取整行
- 每个 `M[row][k]` 被读取 n 次

**Tiled 版本**：

- M 的 Tile 被加载到共享内存
- TILE_WIDTH = 32 个线程共享这个 Tile
- 每个 `M[row][k]` 从全局内存只读 1 次，在共享内存被 32 个线程使用

**计算**：

```text
带宽减少倍数 = TILE_WIDTH = 32
```

N 矩阵同理，也减少 32 倍。

---

### 练习 6

**题目：** 假设一个 CUDA kernel 以 1000 个线程块启动，每块 512 个线程。如果在 kernel 中声明一个局部变量，在执行过程中会创建多少个该变量的版本？

**解答：** **512,000 个**

**分析**：

- 局部变量是每线程私有的
- 总线程数 = 1000 × 512 = 512,000
- 每个线程有自己的变量副本

```cuda
__global__ void kernel() {
    int localVar;  // 512,000 个版本
}
```

---

### 练习 7

**题目：** 在上一题中，如果变量声明为共享内存变量，会创建多少个版本？

**解答：** **1,000 个**

**分析**：

- 共享内存变量是每 Block 一份
- Block 数量 = 1000
- 每个 Block 有一个变量副本

```cuda
__global__ void kernel() {
    __shared__ int sharedVar;  // 1,000 个版本
}
```

---

### 练习 8

**题目：** 考虑两个 N×N 输入矩阵的乘法。当以下情况时，每个输入矩阵的元素被从全局内存请求多少次？

**a. 没有 Tiling？**

**解答：** **N 次**

每个 `M[i][k]` 被 P 第 i 行的所有 N 个元素的计算所使用。由于没有 Tiling，每次使用都从全局内存读取。

---

**b. 使用 T×T 的 Tile？**

**解答：** **N/T 次**

每个元素被加载到共享内存后，被 T 个线程共享使用。原本需要读取 N 次，现在每 T 次计算只需 1 次全局内存读取。

```text
访问次数 = N / T
减少倍数 = T
```

---

### 练习 9

**题目：** 一个 kernel 每线程执行 36 个浮点运算和 7 次 32 位全局内存访问。对于以下设备属性，判断该 kernel 是计算密集型还是内存密集型。

---

#### a. 峰值 FLOPS = 200 GFLOPS，峰值内存带宽 = 100 GB/s

**解答：** **内存密集型（Memory-bound）**

**计算**：

```text
计算限制: 200×10⁹ / 36 = 5.55×10⁹ 次/秒
内存限制: 100×10⁹ / (7×4) = 3.57×10⁹ 次/秒
```

内存限制更严格，所以是内存密集型。

---

#### b. 峰值 FLOPS = 300 GFLOPS，峰值内存带宽 = 250 GB/s

**解答：** **计算密集型（Compute-bound）**

**计算**：

```text
计算限制: 300×10⁹ / 36 = 8.33×10⁹ 次/秒
内存限制: 250×10⁹ / (7×4) = 8.93×10⁹ 次/秒
```

计算限制更严格，所以是计算密集型。

---

### 练习 10

**题目：** 为了操作 Tile，一位新手 CUDA 程序员编写了以下 kernel 来转置矩阵中的每个 Tile：

```cuda
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void BlockTranspose(float* A_elements, int A_width, int A_height) {
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];  // 加载

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];  // 转置写回
}
```

---

**a. 对于 BLOCK_SIZE 的可能取值范围（1 到 20），哪些值能使 kernel 正确执行？**

**解答：** **只有 BLOCK_SIZE = 1**

当 BLOCK_SIZE = 1 时，每个 Block 只有一个线程，不需要同步。对于其他值，由于缺少同步，结果不正确。

---

**b. 如果代码不能对所有 BLOCK_SIZE 正确执行，根本原因是什么？提出修复方案。**

**解答：**

**根本原因**：第 10 行（加载）和第 11 行（写回）之间缺少 `__syncthreads()`。

某些线程可能在其他线程完成加载之前就开始读取 `blockA`，导致读到未初始化的数据。

**修复方案**：

```cuda
__global__ void BlockTranspose(float* A_elements, int A_width, int A_height) {
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    __syncthreads();  // 添加同步！

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

---

### 练习 11

**题目：** 考虑以下 CUDA kernel 和调用它的主机函数：

```cuda
__global__ void foo_kernel(float* a, float* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for(unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if(threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
            + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}
void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<<(N + 128 - 1)/128, 128>>>(a_d, b_d);
}
```

---

**a. 变量 i 有多少个版本？**

**解答：** **1024 个**

- 块数 = (1024 + 127) / 128 = 8
- 每块线程 = 128
- 总线程 = 8 × 128 = 1024
- `i` 是局部变量，每线程一份

---

**b. 数组 x[] 有多少个版本？**

**解答：** **1024 个**

`x[]` 是局部数组，每线程一份。

---

**c. 变量 y_s 有多少个版本？**

**解答：** **8 个**

`y_s` 是共享变量，每 Block 一份，共 8 个 Block。

---

**d. 数组 b_s[] 有多少个版本？**

**解答：** **8 个**

`b_s[]` 是共享数组，每 Block 一份。

---

**e. 每 Block 使用多少共享内存（字节）？**

**解答：** **516 字节**

```text
y_s: 1 × 4 = 4 bytes
b_s[128]: 128 × 4 = 512 bytes
总计: 4 + 512 = 516 bytes
```

---

**f. kernel 的浮点运算与全局内存访问比（OP/B）是多少？**

**解答：** **0.35 OP/B**

**全局内存访问**：

- 读 `a`：4 次 × 4 bytes = 16 bytes
- 读 `b`：1 次 × 4 bytes = 4 bytes
- 写 `b`：1 次 × 4 bytes = 4 bytes
- 总计：24 bytes

**浮点运算**（第 14-15 行）：

- 4 次乘法：`2.5f*x[0]`, `3.7f*x[1]`, `6.3f*x[2]`, `8.5f*x[3]`
- 1 次乘法：`y_s*b_s[threadIdx.x]`
- 5 次加法
- 总计：10 FLOPs（只计算涉及全局内存数据的操作）

如果只计算与读取相关的操作（不含写入）：

```text
比值 = 7 / 20 = 0.35 OP/B
```

---

### 练习 12

**题目：** 考虑一个 GPU 具有以下硬件限制：每 SM 2048 线程、32 块、64K（65536）寄存器、96 KB 共享内存。对于以下 kernel 特性，判断能否达到完全占用率。如果不能，指出限制因素。

---

#### a. 每块 64 线程，每线程 27 个寄存器，每块 4 KB 共享内存

**解答：** ❌ **占用率 75%**，限制因素：**共享内存**

**分析**：

- 线程限制：2048 / 64 = 32 块（满足 ≤32）
- 寄存器限制：32 × 64 × 27 = 55,296（< 65,536，满足）
- 共享内存限制：96 KB / 4 KB = 24 块

实际块数 = min(32, 24) = 24 块

```text
占用率 = 24 × 64 / 2048 = 75%
```

---

#### b. 每块 256 线程，每线程 31 个寄存器，每块 8 KB 共享内存

**解答：** ✅ **占用率 100%**

**分析**：

- 线程限制：2048 / 256 = 8 块（满足 ≤32）
- 寄存器限制：8 × 256 × 31 = 63,488（< 65,536，满足）
- 共享内存限制：96 KB / 8 KB = 12 块（> 8，满足）

实际块数 = 8 块

```text
占用率 = 8 × 256 / 2048 = 100%
```

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0 或更高版本
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（计算能力 3.5+）

## 💡 学习建议

1. **理解 Tiling 本质**：分块加载，块内复用
2. **掌握同步时机**：两次 `__syncthreads()` 缺一不可
3. **分析内存模式**：识别全局内存访问瓶颈
4. **计算算术强度**：FLOP/Byte 决定优化方向
5. **避免 Bank 冲突**：必要时使用 Padding

## 🚀 下一步

完成本章学习后，继续学习：

- 第六章：性能考虑
- 第七章：卷积
- 第八章：模板

---

## 📚 参考资料

- PMPP 第四版 Chapter 5
- [第五章：内存架构和数据局部性](https://smarter.xin/posts/3bb3179b/)

**学习愉快！** 🎓
