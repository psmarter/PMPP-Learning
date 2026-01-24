# 第十一章：前缀和（扫描）

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章系统梳理前缀和（扫描）操作及其 CUDA 优化技术：

- Kogge-Stone 扫描算法
- Brent-Kung 扫描算法
- 控制分歧分析
- 工作效率分析
- 分层扫描（支持任意长度）

**相关博客笔记**：[第十一章：前缀和](https://smarter.xin/posts/a6fc4cf6/)

---

## 💻 代码实现

### Exercise01 - 扫描实现

实现多种扫描 kernel，对应书中图11.3、11.4。

**代码位置**：`Exercise01/`

**实现列表**：

| 实现 | 书中对应 | 特点 |
| ---- | -------- | ---- |
| `scan_sequential` | - | CPU参考实现 |
| `scan_kogge_stone` | 图11.3 | 工作量 O(N log N) |
| `scan_kogge_stone_double_buffer` | 练习2 | 双缓冲消除竞争 |
| `scan_brent_kung` | 图11.4 | 工作量 O(N) |
| `scan_three_phase` | - | 粗化策略 |
| `scan_hierarchical` | 图11.9 | 支持任意长度 |
| `scan_hierarchical_domino` | - | 单kernel，Domino-style同步 |

**核心代码**：

```cuda
// Kogge-Stone 扫描核心思想（图11.3）
__global__ void kogge_stone_scan_kernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float buffer[];
    unsigned int tid = threadIdx.x;

    buffer[tid] = (tid < N) ? X[tid] : 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (tid >= stride) {
            temp = buffer[tid] + buffer[tid - stride];
        }
        __syncthreads();  // 避免写后读竞争
        if (tid >= stride) {
            buffer[tid] = temp;
        }
    }

    if (tid < N) Y[tid] = buffer[tid];
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
================================================================
  第十一章：前缀和（扫描）
  Prefix Sum (Scan) Operations - Multiple Implementations
================================================================

=== 小规模测试（单 Block，999 元素）===

1. CPU 顺序扫描... 完成
2. Kogge-Stone 扫描 (图11.3)... ✅ 结果正确！
3. Kogge-Stone 双缓冲 (练习2)... ✅ 结果正确！
4. Brent-Kung 扫描 (图11.4)... ✅ 结果正确！
5. 三阶段扫描... ✅ 结果正确！

=== 大规模测试（多 Block，100000 元素）===

6. CPU 顺序扫描... 完成
7. 分层扫描（任意长度）... ✅ 结果正确！
8. Domino-Style 分层扫描（单kernel）... ✅ 结果正确！

【关键概念】
• Kogge-Stone：工作量 O(N log N)，步骤 O(log N)，适合低延迟
• Brent-Kung：工作量 O(N)，更高效，分上扫和下扫两阶段
• 双缓冲：消除写后读竞争，无需第二次 __syncthreads()
• 三阶段：粗化策略，每线程处理多个元素
• 分层扫描：支持任意长度，多 Block 协作
• Domino-Style：单kernel实现，动态块索引分配，原子标志同步

✅ 测试完成！
```

---

## 📖 练习题解答

### 练习 1

**题目：** 考虑数组 [4 6 7 1 2 8 5 2]，使用 Kogge-Stone 算法执行包含前缀扫描。报告每步后数组的中间状态。

**解答：**

![Exercise 1 visualization](exercise1.png)

```text
初始:      [4,  6,  7,  1,  2,  8,  5,  2]

stride=1:  [4, 10, 13,  8,  3, 10, 13,  7]
           (每个元素加上左边1个位置的元素)

stride=2:  [4, 10, 17, 18, 16, 18, 16, 17]
           (每个元素加上左边2个位置的元素)

stride=4:  [4, 10, 17, 18, 20, 28, 33, 35]
           (每个元素加上左边4个位置的元素)

最终结果:  [4, 10, 17, 18, 20, 28, 33, 35] ✅
```

---

### 练习 2

**题目：** 修改图11.3的 Kogge-Stone 并行扫描 kernel，使用双缓冲而不是第二次 `__syncthreads()` 来克服写后读竞争条件。

**解答：**

```cpp
__global__ void kogge_stone_scan_kernel_with_double_buffering(float *X, float *Y, unsigned int N){
    extern __shared__ float shared_mem[];
    float* buffer1 = shared_mem;
    float* buffer2 = &shared_mem[N];

    unsigned int tid = threadIdx.x;

    float *src_buffer = buffer1;
    float *trg_buffer = buffer2;

    if (tid < N){
        src_buffer[tid] = X[tid];
    } else {
        src_buffer[tid] = 0.0;
    }
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        // 无需第二次 __syncthreads()
        if (tid >= stride) {
            trg_buffer[tid] = src_buffer[tid] + src_buffer[tid - stride];
        } else {
            trg_buffer[tid] = src_buffer[tid];
        }
        
        // 交换缓冲区
        float* temp = src_buffer;
        src_buffer = trg_buffer;
        trg_buffer = temp;
    }

    if (tid < N){
        Y[tid] = src_buffer[tid];
    }
}
```

该实现已包含在 `Exercise01/solution.cu` 中。

---

### 练习 3

**题目：** 分析图11.3的 Kogge-Stone 并行扫描 kernel。证明控制分歧仅发生在每个 Block 的第一个warp中，且仅在 stride 值小于等于 warp 大小一半时发生。即对于 warp 大小 32，控制分歧将发生在 stride 值为 1, 2, 4, 8, 16 的5次迭代中。

**解答：**

条件 `threadIdx.x >= stride` 决定哪些线程执行。

**stride = 1**: 只有线程 0 不执行 → 只有 Warp 0 有分歧

**stride = 2**: 线程 0, 1 不执行 → 只有 Warp 0 有分歧

**stride = 4**: 线程 0-3 不执行 → 只有 Warp 0 有分歧

**stride = 8**: 线程 0-7 不执行 → 只有 Warp 0 有分歧

**stride = 16**: 线程 0-15 不执行 → 只有 Warp 0 有分歧

**stride = 32**: 线程 0-31 不执行 → Warp 0 完全不活跃，Warp 1+ 全部活跃 → **无分歧**

**stride ≥ 64**: 类似，只有整个 warp 被跳过，无分歧

结论：**只有前5次迭代（stride=1,2,4,8,16）在 Warp 0 中存在控制分歧**。

---

### 练习 4

**题目：** 对于基于归约树的 Kogge-Stone 扫描 kernel，假设有 2048 个元素。哪个选项最接近将执行的加法操作数？

**解答：**

理论公式：`N×log₂(N) - (N - 1) = 2048 × 11 - 2047 = 20,481`

实际计算（每个 Block 1024 线程，2 个 Block）：

| Stride | 活跃线程/Block |
| ------ | -------------- |
| 1      | 1023           |
| 2      | 1022           |
| 4      | 1020           |
| ...    | ...            |
| 512    | 512            |

每个 Block 总操作：1023 + 1022 + ... + 512 = **9,217**

2 个 Block：9,217 × 2 = 18,434

加上块间传播：18,434 + 1,024 ≈ **19,458 次操作**

---

### 练习 5

**题目：** 考虑数组 [4 6 7 1 2 8 5 2]，使用 Brent-Kung 算法执行包含前缀扫描。报告每步后数组的中间状态。

**解答：**

![Visualization of exercise 5](exercise5.png)

Brent-Kung 分两阶段：

**上扫阶段（归约树）：**

```text
初始:      [4,  6,  7,  1,  2,  8,  5,  2]
stride=1:  [4, 10,  7,  8,  2, 10,  5,  7]  (索引1,3,5,7更新)
stride=2:  [4, 10,  7, 18,  2, 10,  5, 17] (索引3,7更新)
stride=4:  [4, 10,  7, 18,  2, 10,  5, 35] (索引7更新)
```

**下扫阶段（分发树）：**

```text
stride=2:  [4, 10,  7, 18,  2, 28,  5, 35] (索引5更新)
stride=1:  [4, 10, 17, 18, 20, 28, 33, 35] (索引2,4,6更新)
```

最终结果：**[4, 10, 17, 18, 20, 28, 33, 35]** ✅

---

### 练习 6

**题目：** 对于 Brent-Kung 扫描 kernel，假设有 2048 个元素。在归约树阶段和逆归约树阶段分别执行多少次加法操作？

**解答：**

Brent-Kung 每个线程处理2个元素，2048元素只需1个 Block，1024线程。

**归约树阶段（上扫）：**

| Stride | 活跃线程 |
| ------ | -------- |
| 1      | 1024     |
| 2      | 512      |
| 4      | 256      |
| ...    | ...      |
| 1024   | 1        |

总计：1024 + 512 + ... + 1 = **2,047 次操作**

**逆归约树阶段（下扫）：**

| Stride | 活跃线程 |
| ------ | -------- |
| 512    | 1        |
| 256    | 3        |
| 128    | 7        |
| ...    | ...      |
| 1      | 1019     |

总计：1 + 3 + 7 + 15 + 31 + 63 + 127 + 255 + 509 + 1019 ≈ **2,030 次操作**

**总计：2,047 + 2,030 = 4,077 次操作**

理论公式：`2N - 2 - log₂(N) = 2×2048 - 2 - 11 = 4,083`，非常接近！

---

### 练习 7

**题目：** 使用图11.4中的算法完成排他扫描 kernel。

**解答：**

![alt text](exercise7.png)

排他扫描与包含扫描的区别：输出的第 i 个元素是输入前 i 个元素（不包括第 i 个）的和。

只需在最后输出时做调整：

```cpp
// 包含扫描转排他扫描
if (tid < N) {
    if (tid == 0) {
        Y[tid] = 0;  // 排他扫描第一个元素为0
    } else {
        Y[tid] = sdata[tid - 1];  // 前一个元素的包含扫描结果
    }
}
```

---

### 练习 8

**题目：** 完成图11.9分段并行扫描算法的主机代码和所有三个 kernel。

**解答：**

分层扫描分三个阶段：

1. **阶段1**：每个 Block 执行局部 Kogge-Stone 扫描，保存块末尾和
2. **阶段2**：对块末尾和数组执行扫描（递归或单 Block）
3. **阶段3**：将扫描后的块和加到各块元素上

该实现已包含在 `Exercise01/solution.cu` 的 `scan_hierarchical` 函数中。

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0 或更高版本
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（计算能力 3.5+）

## 💡 学习建议

1. **理解两种算法**：Kogge-Stone（低延迟）vs Brent-Kung（高效率）
2. **双缓冲技术**：消除写后读竞争的通用方法
3. **工作效率分析**：O(N log N) vs O(N) 的实际影响
4. **分层处理**：处理超出单 Block 容量的数据
5. **控制分歧**：只在 stride < warp_size 时存在

## 🚀 下一步

完成本章学习后，继续学习：

- 第十二章：合并
- 第十三章：排序
- 第十四章：稀疏矩阵计算

---

## 📚 参考资料

- PMPP 第四版 Chapter 11
- [第十一章：前缀和](https://smarter.xin/posts/a6fc4cf6/)

**学习愉快！** 🎓
