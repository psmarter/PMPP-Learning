# 第十三章：排序

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章系统梳理 GPU 排序算法及其 CUDA 优化技术：

- 基数排序原理（按位排序、前缀和）
- 内存访问优化（合并写入）
- 多位基数优化（减少迭代次数）
- 线程粗化（提高效率）
- 并行归并排序

**相关博客笔记**：[第十三章：排序](https://smarter.xin/posts/d9ee9484/)

---

## 💻 代码实现

### Exercise01 - 排序实现

实现5种排序 kernel，对应书中图13.4及优化版本。

**代码位置**：`Exercise01/`

**实现列表**：

| 实现 | 书中对应 | 特点 |
| ---- | -------- | ---- |
| `gpuRadixSortNaive` | 图13.4 | 朴素三核：提取位、扫描、分散 |
| `gpuRadixSortCoalesced` | 练习1 | 共享内存优化，内存合并写入 |
| `gpuRadixSortMultibit` | 练习2 | 多位基数（4位/轮） |
| `gpuRadixSortCoarsened` | 练习3 | 线程粗化（每线程4元素） |
| `gpuMergeSort` | 练习4 | 并行归并排序 |

**核心代码**：

```cuda
// 基数排序核心：分散（scatter）
__global__ void scatterKernel(unsigned int* input, unsigned int* output, 
                              unsigned int* scannedBits, int N, int iter, unsigned int totalOnes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int key = input[tid];
        unsigned int bit = (key >> iter) & 1;
        unsigned int numOnesBefore = scannedBits[tid];
        
        // 0放前面，1放后面
        unsigned int dst = (bit == 0) ? (tid - numOnesBefore) : (N - totalOnes + numOnesBefore);
        output[dst] = key;
    }
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
  第十三章：排序
  Parallel Sorting - Multiple Implementations
================================================================

配置:
  数组长度: 1000000
  BLOCK_SIZE: 1024
  RADIX_BITS: 4 (桶数: 16)
  COARSE_FACTOR: 4

=== 正确性验证 ===

1. 朴素三核基数排序 (书中图13.4)... ✅ 结果正确！
2. 内存合并基数排序 (练习1: 共享内存优化)... ✅ 结果正确！
3. 多位基数排序 (练习2: 4位/轮)... ✅ 结果正确！
4. 线程粗化基数排序 (练习3: 每线程4元素)... ✅ 结果正确！
5. 并行归并排序 (练习4: 使用第12章归并)... ✅ 结果正确！
```

---

## 📖 练习题解答

### 练习 1

**题目：** 扩展图13.4中的 kernel，使用共享内存改进内存合并。

**解答：**

核心优化：分开计算0和1的偏移，实现连续内存写入。

```cuda
// 本地扫描 kernel - 在共享内存中执行 Blelloch 扫描
__global__ void localScanKernel(unsigned int* d_input, unsigned int* d_localScan, 
                                unsigned int* d_blockOneCount, int N, int iter) {
    extern __shared__ unsigned int s_bits[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 加载到共享内存
    unsigned int bit_val = (gid < N) ? ((d_input[gid] >> iter) & 1) : 0;
    s_bits[tid] = bit_val;
    __syncthreads();
    
    // Blelloch 上扫阶段
    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            s_bits[index] += s_bits[index - offset];
        }
        __syncthreads();
    }
    
    // 保存总和并清零
    if (tid == 0) {
        d_blockOneCount[blockIdx.x] = s_bits[blockDim.x - 1];
        s_bits[blockDim.x - 1] = 0;
    }
    __syncthreads();
    
    // Blelloch 下扫阶段
    for (unsigned int offset = blockDim.x / 2; offset >= 1; offset /= 2) {
        int index = (tid + 1) * offset * 2 - 1;
        if (index < blockDim.x) {
            unsigned int t = s_bits[index - offset];
            s_bits[index - offset] = s_bits[index];
            s_bits[index] += t;
        }
        __syncthreads();
        if (offset == 1) break;
    }
    
    if (gid < N) d_localScan[gid] = s_bits[tid];
}
```

优化效果：减少全局内存访问，提高写入带宽利用率。

---

### 练习 2

**题目：** 扩展图13.4中的 kernel，处理多位基数。

**解答：**

每次处理4位（16个桶），32位整数只需8轮迭代（vs 32轮）。

```cuda
// 多位基数：使用直方图统计
__global__ void localScanKernelMultibit(const unsigned int* d_input, unsigned int* d_localOffsets,
                                        unsigned int* d_blockHist, int N, int iter, int r) {
    const unsigned int numBuckets = 1 << r;  // 2^r 个桶
    extern __shared__ unsigned int shared[];
    
    unsigned int* s_hist = shared;
    unsigned int* s_digits = &s_hist[numBuckets];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // 提取当前位组的数字（4位 = 0~15）
    unsigned int digit = 0;
    if (gid < N) {
        unsigned int key = d_input[gid];
        digit = (key >> (iter * r)) & (numBuckets - 1);
    }
    s_digits[tid] = digit;
    
    // 初始化并构建直方图
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();
    
    if (gid < N) atomicAdd(&s_hist[digit], 1);
    __syncthreads();
    
    // 写入 Block 直方图
    for (unsigned int i = tid; i < numBuckets; i += blockDim.x)
        d_blockHist[blockIdx.x * numBuckets + i] = s_hist[i];
    
    // ... 计算本地偏移 ...
}
```

优化效果：迭代次数从32轮减少到8轮。

---

### 练习 3

**题目：** 扩展图13.4中的 kernel，应用线程粗化改进内存合并。

**解答：**

每个线程处理 COARSE_FACTOR（如4个）元素。

```cuda
__global__ void localScanKernelCoarsened(...) {
    int tid = threadIdx.x;
    int baseIdx = blockIdx.x * blockDim.x * COARSE_FACTOR + tid * COARSE_FACTOR;
    
    // 每个线程处理多个元素
    for (int i = 0; i < COARSE_FACTOR; i++) {
        int idx = baseIdx + i;
        unsigned int digit = 0;
        if (idx < N) {
            unsigned int key = d_input[idx];
            digit = (key >> (iter * r)) & (numBuckets - 1);
        }
        s_digits[tid * COARSE_FACTOR + i] = digit;
    }
    // ...
}
```

优化效果：减少线程数量，更好的寄存器利用率。

---

### 练习 4

**题目：** 使用第12章的并行归并实现并行归并排序。

**解答：**

两阶段：Block 内隐式排序 + 跨 Block 归并。

```cuda
// 归并一轮：每个 Block 归并一对相邻有序段
__global__ void mergePassKernel(unsigned int* d_in, unsigned int* d_out, int N, int width) {
    int pair = blockIdx.x;
    int start = pair * (2 * width);
    if (start >= N) return;
    
    int mid = min(start + width, N);
    int end = min(start + 2 * width, N);
    
    unsigned int* A = d_in + start;
    unsigned int* B = d_in + mid;
    unsigned int* C = d_out + start;
    
    // 使用 co-rank 分配工作
    int tid = threadIdx.x;
    int k_start = tid * elementsPerThread;
    int i_start = co_rank(k_start, A, lenA, B, lenB);
    int j_start = k_start - i_start;
    
    // 顺序归并
    merge_sequential(A + i_start, ..., B + j_start, ..., C + k_start);
}

// 主函数：多轮归并
void gpuMergeSort(unsigned int* d_input, int N) {
    int width = 1;
    while (width < N) {
        int numMerges = (N + 2 * width - 1) / (2 * width);
        mergePassKernel<<<numMerges, BLOCK_SIZE>>>(d_input, d_output, N, width);
        swap(d_input, d_output);
        width *= 2;
    }
}
```

优化效果：适合任意可比较类型，O(n log n) 复杂度。

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0 或更高版本
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（计算能力 3.5+）

## 💡 学习建议

1. **理解基数排序**：按位处理，利用前缀和确定输出位置
2. **内存优化**：合并写入减少内存事务数
3. **多位处理**：减少迭代次数的代价是更多的桶
4. **线程粗化**：权衡并行度和每线程工作量
5. **实际应用场景**：基数排序适合整数排序，对于浮点数需要先转换为整数；在生产环境中，考虑使用 Thrust 库的 `thrust::sort` 或 cuSort，它们针对不同数据类型和规模进行了高度优化

## 🚀 下一步

完成本章学习后，继续学习：

- 第十四章：稀疏矩阵计算
- 第十五章：图遍历
- 第十六章：深度学习

---

## 📚 参考资料

- PMPP 第四版 Chapter 13
- [第十三章：排序](https://smarter.xin/posts/d9ee9484/)

**学习愉快！** 🎓
