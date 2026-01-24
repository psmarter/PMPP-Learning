# 第二章：异构数据并行计算

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章介绍了 CUDA 编程的基础知识，包括：

- CUDA 程序结构
- 线程、块、网格的组织方式
- 内存分配和数据传输
- Kernel 函数的编写和启动
- 基本的向量运算实现

**相关博客笔记**：[第二章：异构数据并行计算](https://smarter.xin/posts/3ee22ce5/)

---

## 💻 代码实现

### Exercise01 - 向量乘法

实现了一个基本的**向量乘法（Vector Multiplication）** kernel。

**代码位置**：[Exercise01/solution.cu](Exercise01/solution.cu)

**功能**：`C[i] = A[i] * B[i]`

#### 运行方法

```bash
# 进入练习目录
cd Exercise01

# 编译
make

# 运行测试
make run

# 清理
make clean
```

#### 预期输出

```
Device 0: NVIDIA GeForce RTX 4090
  Compute Capability: 8.9
  Total Global Memory: 24.00 GB

=== Correctness Test ===
Testing vector multiplication with 1048576 elements...
✅ Correctness test PASSED!

=== Performance Test ===
Data size: 1048576 elements (4.00 MB)
Iterations: 100

Results:
  Average time per iteration: 0.125 ms
  Effective bandwidth: 96.00 GB/s

✅ All tests completed successfully!
```

---

## 📖 练习题解答

### 练习 1

**题目：** 如果我们想使用网格中的每个线程来计算向量加法的一个输出元素，那么将线程/块索引映射到数据索引 (i) 的表达式应该是什么？

**选项：**

- A. `i = threadIdx.x + threadIdx.y;`
- B. `i = blockIdx.x + threadIdx.x;`
- C. `i = blockIdx.x * blockDim.x + threadIdx.x;`
- D. `i = blockIdx.x * threadIdx.x;`

**解答：**
  
**C** 正确答案是 `i = blockIdx.x * blockDim.x + threadIdx.x`。

**解释**：需要计算线程的全局索引。每个块有 `blockDim.x` 个线程，块 ID 为 `blockIdx.x`，块内线程 ID 为 `threadIdx.x`。

**示例**：假设每个块有 256 个线程，访问块 1 中的第 128 个线程：

- `i = 1 * 256 + 128 = 384` ✅

---

### 练习 2

**题目：** 假设我们想使用每个线程来计算向量加法的两个相邻元素。那么将线程/块索引映射到该线程要处理的第一个元素的数据索引 (i) 的表达式应该是什么？

**选项：**

- A. `i = blockIdx.x * blockDim.x + threadIdx.x * 2;`
- B. `i = blockIdx.x * threadIdx.x * 2;`
- C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**解答：**

**C** `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2`

**解释**：每个线程处理两个相邻元素，例如线程 0 处理 (0, 1)，线程 1 处理 (2, 3)，线程 2 处理 (4, 5)...

**示例计算**：

- 块 0，线程 0：`(0 * 256 + 0) * 2 = 0` → 处理元素 0, 1
- 块 0，线程 1：`(0 * 256 + 1) * 2 = 2` → 处理元素 2, 3
- 块 4，线程 0：`(4 * 256 + 0) * 2 = 2048` → 处理元素 2048, 2049

---

### 练习 3

**题目：** 我们想使用每个线程来计算向量加法的两个元素。每个线程块处理 `2 * blockDim.x` 个连续元素，这些元素形成两个部分。块中的所有线程首先处理第一个部分，每个线程处理一个元素。然后它们都移动到下一个部分，每个线程再处理一个元素。将线程/块索引映射到第一个元素的数据索引的表达式应该是什么？

**选项：**

- A. `i = blockIdx.x * blockDim.x + threadIdx.x + 2;`
- B. `i = blockIdx.x * threadIdx.x * 2;`
- C. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;`
- D. `i = blockIdx.x * blockDim.x * 2 + threadIdx.x;`

**解答：**

**D** `i = blockIdx.x * blockDim.x * 2 + threadIdx.x`

**解释**：每个块处理 `2 × 块大小` 个元素，分为两部分。

**示例**（假设块大小为 256）：

- **块 0**：处理元素 0-511
  - 第一部分（0-255）：线程 0 处理元素 0，线程 1 处理元素 1...
  - 第二部分（256-511）：线程 0 处理元素 256，线程 1 处理元素 257...
- **块 1**：处理元素 512-1023
  - 第一部分（512-767）：从 `1 * 256 * 2 + 0 = 512` 开始

线程处理的索引对：`(0, 256)`, `(1, 257)`, `(2, 258)` ...

---

### 练习 4

**题目：** 对于向量加法，假设向量长度为 8000，每个线程计算一个输出元素，线程块大小为 1024 个线程。程序员配置 kernel 调用使用最少数量的线程块来覆盖所有输出元素。网格中将有多少个线程？

**选项：**

- A. 8000
- B. 8196
- C. 8192
- D. 8200

**解答：**

**C. 8192**

**计算**：

- 需要的块数：`⌈8000 / 1024⌉ = 8` 个块
- 总线程数：`8 × 1024 = 8192` 个线程
- 实际使用：8000 个线程（最后 192 个线程空闲）

---

### 练习 5

**题目：** 如果我们想在 CUDA 设备全局内存中分配一个包含 v 个整数元素的数组，那么 `cudaMalloc` 调用的第二个参数应该使用什么表达式？

**选项：**

- A. `n`
- B. `v`
- C. `n * sizeof(int)`
- D. `v * sizeof(int)`

**解答：**

**D. `v * sizeof(int)`**

**解释**：`cudaMalloc` 的第二个参数是**字节数**。

```cuda
int *d_array;
cudaMalloc(&d_array, v * sizeof(int));  // v 个整数 × 每个整数的字节数
```

---

### 练习 6

**题目：** 如果我们想分配一个包含 n 个浮点元素的数组，并让浮点指针变量 `A_d` 指向分配的内存，那么 `cudaMalloc` 调用的第一个参数应该使用什么表达式？

**选项：**

- A. `n`
- B. `(void*) A_d`
- C. `*A_d`
- D. `(void**) &A_d`

**解答：**

**D. `(void**) &A_d`**

**解释**：`cudaMalloc` 需要**指向指针的指针**（二级指针）。

```cuda
float *A_d;  // 这是一个指针
cudaMalloc((void**)&A_d, n * sizeof(float));
//         ^^^^^^^ 二级指针：指向 A_d 的地址
```

- `A_d` 是指针
- `&A_d` 是指针的地址（指向指针的指针）
- `(void**)` 类型转换为 void**

---

### 练习 7

**题目：** 如果我们想从主机数组 `A_h`（指向源数组元素 0 的指针）复制 3000 字节的数据到设备数组 `A_d`（指向目标数组元素 0 的指针），那么在 CUDA 中进行此数据复制的适当 API 调用是什么？

**选项：**

- A. `cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);`
- B. `cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceToHost);`
- C. `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`
- D. `cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);`

**解答：**

**C. `cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);`**

**解释**：`cudaMemcpy` 参数顺序：

```cuda
cudaMemcpy(目标, 源, 字节数, 方向);
cudaMemcpy(dst,  src, size,  direction);
```

- **目标**：`A_d`（设备）
- **源**：`A_h`（主机）
- **字节数**：3000
- **方向**：`cudaMemcpyHostToDevice`（主机→设备）

---

### 练习 8

**题目：** 如何声明一个变量 `err` 来适当地接收 CUDA API 调用的返回值？

**选项：**

- A. `int err;`
- B. `cudaError err;`
- C. `cudaError_t err;`
- D. `cudaSuccess_t err;`

**解答：**

**C. `cudaError_t err;`**

**使用示例**：

```cuda
cudaError_t err;

err = cudaMalloc(&d_data, size);
if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
}
```

---

### 练习 9

考虑以下 CUDA kernel 和调用它的相应主机函数：

```c
01 __global__ void foo_kernel(float* a, float* b, unsigned int N) {
02     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
03     
04     if (i < N) {
05         b[i] = 2.7f * a[i] - 4.3f;
06     }
07 }
08 
09 void foo(float* a_d, float* b_d) {
10     unsigned int N = 200000;
11     foo_kernel<<<(N + 128 - 1) / 128, 128>>>(a_d, b_d, N);
12 }
```

#### a. 每个块有多少个线程？

**答案：128 个**

由 kernel 启动参数的第二个参数指定：`<<<gridSize, 128>>>`

#### b. 网格中有多少个线程？

**答案：200064 个**

**计算**：

- 块数：`(200000 + 128 - 1) / 128 = 1563` 个块
- 总线程数：`1563 × 128 = 200064` 个线程

#### c. 网格中有多少个块？

**答案：1563 个**

如上所示：`(200000 + 127) / 128 = 1563`

#### d. 有多少个线程执行第 02 行的代码？

**答案：200064 个**

所有线程都会执行线程索引计算（第 02 行）。

#### e. 有多少个线程执行第 05 行的代码？

**答案：200000 个**

**解释**：

- 第 04 行有边界检查：`if (i < N)`
- 只有 `i < 200000` 的线程会执行第 05 行
- 最后 64 个线程（索引 200000-200063）不会执行

---

### 练习 10

**题目：** 一个新来的暑期实习生对 CUDA 感到沮丧。他一直抱怨 CUDA 非常繁琐。他必须将许多计划在主机和设备上执行的函数声明两次，一次作为主机函数，一次作为设备函数。你的回应是什么？

**解答：**

可以使用 `__host__` 和 `__device__` 函数类型限定符来同时声明主机和设备版本：

```cuda
// 同时编译为主机和设备函数
__host__ __device__
float myFunction(float x) {
    return x * x + 2.0f * x + 1.0f;
}

// 现在可以在主机和设备代码中都调用
```

CUDA 编译器会自动编译该函数的两个版本，无需重复代码。

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0 或更高版本
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（计算能力 3.5+）

## 📝 代码结构

```
Exercise01/
├── solution.h          # 函数声明头文件
├── solution.cu         # CUDA kernel 和 host 函数实现
├── test.cpp            # 测试代码（正确性验证 + 性能测试）
├── Makefile            # 编译脚本
└── ../../Common/       # 公共工具库
    ├── utils.cuh       # CUDA 错误检查宏
    └── timer.h         # 性能计时器（CPU + CUDA）
```

## 💡 学习建议

1. **理解基础**：先理解线程索引计算公式
2. **动手实践**：修改 Exercise01 代码，尝试不同的块大小
3. **解答练习题**：完成上述所有练习题
4. **性能分析**：对比不同配置的性能差异
5. **错误调试**：学习使用 `cuda-memcheck` 检查内存错误

## 🚀 下一步

完成本章学习后，继续学习：

- 第三章：多维网格和数据
- 第四章：内存架构和数据局部性
- 第五章：性能优化技术

---

## 📚 参考资料

- PMPP 第四版 Chapter 2
- [第二章：异构数据并行计算](https://smarter.xin/posts/3ee22ce5/)

**学习愉快！** 🎓
