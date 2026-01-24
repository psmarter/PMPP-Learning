# 第十章：归约

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章系统梳理归约（Reduction）操作及其 CUDA 优化技术：

- 简单归约与控制分歧
- 收敛归约（优化分歧）
- 共享内存优化
- 分段归约（支持任意长度）
- 线程粗化（Thread Coarsening）

**相关博客笔记**：[第十章：归约和最小化发散](https://smarter.xin/posts/43b40d12/)

---

## 💻 代码实现

### Exercise01 - 归约实现

实现多种归约 kernel，对应书中图10.6、10.9、10.11、10.15。

**代码位置**：`Exercise01/`

**实现列表**：

| 实现 | 书中对应 | 特点 |
| ---- | -------- | ---- |
| `reduction_sequential` | - | CPU参考实现 |
| `reduction_simple` | 图10.6 | 简单归约，分歧严重 |
| `reduction_convergent` | 图10.9 | 收敛归约，消除分歧 |
| `reduction_convergent_reversed` | 练习3 | 反向收敛归约 |
| `reduction_shared_memory` | 图10.11 | 共享内存归约 |
| `reduction_segmented` | - | 分段归约，支持任意长度 |
| `reduction_coarsened` | 图10.15 | 线程粗化归约 |
| `max_reduction_coarsened` | 练习4 | 最大值归约 |

**核心代码**：

```cuda
// 收敛归约核心思想（图10.9）
__global__ void convergent_sum_reduction_kernel(float* input, float* output) {
    unsigned int i = threadIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *output = input[0];
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
  第十章：归约
  Reduction Operations - Multiple Implementations
================================================================

=== 小规模测试（单 Block，2048 元素）===

1. CPU 顺序归约... 结果: 2048.00
2. 简单归约 (图10.6)... 结果: 2048.00 ✅ 正确！
3. 收敛归约 (图10.9)... 结果: 2048.00 ✅ 正确！
4. 反向收敛归约 (练习3)... 结果: 2048.00 ✅ 正确！
5. 共享内存归约 (图10.11)... 结果: 2048.00 ✅ 正确！

=== 大规模测试（多 Block，10000000 元素）===

6. CPU 顺序归约... 结果: 10000000.00
7. 分段归约... 结果: 10000000.00 ✅ 正确！
8. 线程粗化归约 (图10.15)... 结果: 10000000.00 ✅ 正确！
```

---

## 📖 练习题解答

### 练习 1

**题目：** 对于图10.6的简单归约 kernel，如果元素数为1024，warp 大小为32，第5次迭代时有多少个 warp 存在分歧？

**解答：**

图10.6 的简单归约 kernel：

```cpp
__global__ void simple_sum_reduction_kernel(float* input, float* output){
    unsigned int i = 2 * threadIdx.x;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        if (threadIdx.x % stride == 0){
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        *output = input[0];
}
```

1024 元素，每个线程处理2个元素，所以有 `1024/2 = 512` 个线程。

512 线程 / 32 = **16 个 warp**。

各迭代的 stride：

- 迭代 1: stride = 1
- 迭代 2: stride = 2
- 迭代 3: stride = 4
- 迭代 4: stride = 8
- 迭代 5: stride = 16

第5次迭代时，满足 `threadIdx.x % 16 == 0` 的线程执行：线程 0, 16, 32, 48, ..., 496。

每个 warp（32线程）中有2个线程活跃。例如：

- Warp 0: 线程 0 和 16 活跃
- Warp 15: 线程 480 和 496 活跃

所有 **16 个 warp 都有控制分歧**。

---

### 练习 2

**题目：** 对于图10.9的收敛归约 kernel，如果元素数为1024，warp 大小为32，第5次迭代时有多少个 warp 存在分歧？

**解答：**

图10.9 的收敛归约 kernel：

```cpp
__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        *output = input[0];
    }
}
```

同样 512 线程，16 个 warp。

各迭代的 stride：

- 迭代 1: stride = 512
- 迭代 2: stride = 256
- 迭代 3: stride = 128
- 迭代 4: stride = 64
- 迭代 5: stride = 32

第5次迭代时，只有 `threadIdx.x < 32` 的线程活跃，即线程 0-31。

这正好是 **1 个完整的 warp**（Warp 0），其中所有线程都活跃，**无分歧**。

其他 15 个 warp 完全不活跃，也无分歧。

答案：**0 个 warp 有分歧**。

![10.9 kernel setting visualization](exercise_2_visualization.png)

---

### 练习 3

**题目：** 修改图10.9的 kernel，使用下图所示的访问模式（从右向左收敛）。

![the new kernel visualization](exercise_3_visualization.png)

**解答：**

```cpp
__global__ void convergent_sum_reduction_kernel_reversed(float* input, float* output){
    unsigned int i = threadIdx.x + blockDim.x;
    
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
        // stride 不变，但从右侧索引
        if (blockDim.x - threadIdx.x <= stride){
            input[i] += input[i - stride];
        }
        __syncthreads();
    }
    
    // 结果在最后一个元素
    if (threadIdx.x == blockDim.x - 1){
        *output = input[i];
    }
}
```

该实现已包含在 `Exercise01/solution.cu` 中。

---

### 练习 4

**题目：** 修改图10.15的 kernel，执行最大值归约而不是求和归约。

**解答：**

```cpp
__global__ void CoarsenedMaxReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    
    float maximum_value = input[i];
    for(unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
        maximum_value = fmax(maximum_value, input[i + tile*BLOCK_DIM]);
    }
    input_s[t] = maximum_value;

    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if (t < stride) {
            input_s[t] = fmax(input_s[t], input_s[t + stride]);
        }
    }
    
    if (t == 0) {
        atomicMax((int*)output, __float_as_int(input_s[0]));  // 需要适当处理
    }
}
```

该实现已包含在 `Exercise01/solution.cu` 中。

---

### 练习 5

**题目：** 修改图10.15的 kernel，使其支持任意长度的输入（不必是 `COARSE_FACTOR*2*blockDim.x` 的倍数）。添加参数 N 表示输入长度。

**解答：**

```cpp
__global__ void coarsened_sum_reduction_kernel(float* input, float* output, int length){
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    
    float sum = 0.0f;
    // 只在数组范围内累加
    if (i < length){
        sum = input[i];
    
        for(unsigned int tile = 1; tile < COARSE_FACTOR*2; ++tile) {
            // 只累加数组范围内的元素
            if (i + tile*BLOCK_DIM < length) {
                sum += input[i + tile*BLOCK_DIM];
            }
        }
    }

    input_s[t] = sum;
    
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}
```

该实现已包含在 `Exercise01/solution.cu` 中。

---

### 练习 6

**题目：** 假设对以下输入数组进行并行归约：

`[6, 2, 7, 4, 5, 8, 3, 1]`

展示每次迭代后数组内容的变化：

#### a. 使用图10.6的未优化 kernel

![exercise 6a visualization](exercise_6a_visualization.png)

```text
初始:     [6, 2, 7, 4, 5, 8, 3, 1]

迭代 1 (stride=1):
  线程 0: input[0] += input[1]  →  6+2=8
  线程 2: input[2] += input[3]  →  7+4=11
  线程 4: input[4] += input[5]  →  5+8=13  (错误！应该是 input[4]+input[5])
  线程 6: input[6] += input[7]  →  3+1=4
结果:     [8, 2, 11, 4, 13, 8, 4, 1]

迭代 2 (stride=2):
  线程 0: input[0] += input[2]  →  8+11=19
  线程 4: input[4] += input[6]  →  13+4=17
结果:     [19, 2, 11, 4, 17, 8, 4, 1]

迭代 3 (stride=4):
  线程 0: input[0] += input[4]  →  19+17=36
结果:     [36, 2, 11, 4, 17, 8, 4, 1]

最终结果: 36 ✅
```

#### b. 使用图10.9的优化 kernel

![exercise 6b visualization](exercise_6b_visualization.png)

```text
初始:     [6, 2, 7, 4, 5, 8, 3, 1]

迭代 1 (stride=4):
  线程 0: input[0] += input[4]  →  6+5=11
  线程 1: input[1] += input[5]  →  2+8=10
  线程 2: input[2] += input[6]  →  7+3=10
  线程 3: input[3] += input[7]  →  4+1=5
结果:     [11, 10, 10, 5, 5, 8, 3, 1]

迭代 2 (stride=2):
  线程 0: input[0] += input[2]  →  11+10=21
  线程 1: input[1] += input[3]  →  10+5=15
结果:     [21, 15, 10, 5, 5, 8, 3, 1]

迭代 3 (stride=1):
  线程 0: input[0] += input[1]  →  21+15=36
结果:     [36, 15, 10, 5, 5, 8, 3, 1]

最终结果: 36 ✅
```

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0 或更高版本
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（计算能力 3.5+）

## 💡 学习建议

1. **理解控制分歧**：简单归约 vs 收敛归约的分歧差异
2. **共享内存**：减少全局内存访问次数
3. **原子操作**：`atomicAdd` 用于 Block 间结果合并
4. **线程粗化**：每线程处理多元素，减少 Block 开销
5. **边界处理**：支持任意长度输入的关键

## 🚀 下一步

完成本章学习后，继续学习：

- 第十一章：前缀和
- 第十二章：合并
- 第十三章：排序

---

## 📚 参考资料

- PMPP 第四版 Chapter 10
- [第十章：归约和最小化发散](https://smarter.xin/posts/43b40d12/)

**学习愉快！** 🎓
