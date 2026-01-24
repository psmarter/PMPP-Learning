# 第三章：多维网格和数据

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章介绍了多维数据网格的处理，包括：

- 多维线程组织（2D/3D 网格和块）
- 矩阵和图像数据的线程映射
- 行优先和列优先存储顺序
- 图像处理 kernel（RGB 转灰度、高斯模糊）
- 矩阵运算（矩阵乘法、矩阵向量乘法）

**相关博客笔记**：[第三章：多维网格和数据](https://smarter.xin/posts/6b7045b6/)

---

## 💻 代码实现

本章共实现了5个练习，涵盖矩阵运算和图像处理：

### Exercise01 - 矩阵乘法（行级和列级 kernel）

实现了两种不同的矩阵乘法策略，用于对比不同粒度的线程并行方式：

- **行级 kernel (`matrixMulRowKernel`)**：每个线程计算输出矩阵的一整行
- **列级 kernel (`matrixMulColKernel`)**：每个线程计算输出矩阵的一整列

**代码位置**：`Exercise01/`

**性能分析**：

- 行级/列级方法：并行度低（只有 N 个线程），每个线程工作量大
- 适用场景：演示不同并行策略的性能差异
- 不推荐：实际应用中应使用元素级方法（见 Exercise03）

#### 运行 Exercise01

```bash
cd Exercise01
make
make run
```

#### 预期输出

```text
================================================
第三章 - 练习1: 矩阵乘法（行级和列级 kernel）
================================================

Device 0: "NVIDIA GeForce RTX 4090"
  Compute Capability: 8.9
  ...

=== 正确性测试 ===
测试矩阵大小: 64×64

测试行级 kernel...
测试列级 kernel...
计算 CPU 参考结果...

行级 kernel: ✅ 通过
列级 kernel: ✅ 通过

=== 性能测试 ===
矩阵大小: 512×512
迭代次数: 10

行级 kernel:
  平均时间: 45.231 ms
  性能: 5.94 GFLOPS

列级 kernel:
  平均时间: 46.102 ms
  性能: 5.82 GFLOPS

✅ 所有测试完成！
```

---

### Exercise02 - 矩阵向量乘法

实现了矩阵向量乘法：`A = B × C`（A 是向量，B 是矩阵，C 是向量）

**代码位置**：`Exercise02/`

**功能**：每个线程计算输出向量的一个元素

**公式**：`A[i] = sum_over_j(B[i][j] * C[j])`

#### 运行 Exercise02

```bash
cd Exercise02
make
make run
```

#### 预期输出

```text
==================================================
第三章 - 练习2: 矩阵向量乘法
==================================================

=== 正确性测试 ===
矩阵: 128×256, 向量: 256

执行 GPU 计算...
执行 CPU 验证...
✅ 正确性测试通过

=== 性能测试 ===
矩阵大小: 1024×2048
迭代次数: 100

结果:
  平均时间: 0.156 ms
  性能: 26.78 GFLOPS

✅ 所有测试完成！
```

---

### Exercise03 - 标准矩阵乘法（元素级 kernel）

实现了最常用的矩阵乘法方式：每个线程计算输出矩阵的一个元素。

**代码位置**：`Exercise03/`

**特点**：

- 使用 2D 线程块和网格组织
- 充分利用 GPU 并行性（N² 个线程）
- 支持非方阵乘法（M×N 和 N×O）

**这是推荐的标准实现方式！**

#### 运行 Exercise03

```bash
cd Exercise03
make
make run
```

#### 预期输出

```text
==================================================
第三章 - 练习3: 标准矩阵乘法（元素级 kernel)
==================================================

=== 正确性测试 ===
测试: M(64×128) × N(128×96) = P(64×96)

执行 GPU 计算...
执行 CPU 验证...
✅ 正确性测试通过

=== 性能测试 ===
矩阵大小: 512×512
迭代次数: 20

结果:
  平均时间: 2.134 ms
  性能: 127.56 GFLOPS

✅ 所有测试完成！
```

---

### Exercise04 - RGB 转灰度

将彩色图像（RGB 三通道）转换为灰度图像（单通道）。

**代码位置**：`Exercise04/`

**转换公式**：`Gray = 0.21*R + 0.71*G + 0.07*B`

**特点**：

- 使用 2D 线程组织，每个线程处理一个像素
- **使用真实图像测试**：Grace Hopper 的经典照片
- 使用 stb_image 单头文件库（无需额外依赖）
- 自动保存输出图像用于对比

**输出**：

- `output_gpu.jpg`：GPU 计算结果
- `output_cpu.jpg`：CPU 验证结果

#### 运行 Exercise04

```bash
cd Exercise04
make
make run
```

#### 预期输出

```text
==================================================
第三章 - 练习: RGB 转灰度
==================================================

已加载图像: Grace_Hopper.jpg
图像尺寸: 600×600, 通道数: 3

=== GPU 处理 ===
转换完成，用时: 0.234 ms
已保存: output_gpu.jpg

=== CPU 验证 ===
转换完成，用时: 12.456 ms
已保存: output_cpu.jpg

=== 结果验证 ===
✅ GPU 和 CPU 结果完全一致

加速比: 53.23x
✅ 测试完成！
```

**输出文件**：`output_gpu.jpg`, `output_cpu.jpg`

---

### Exercise05 - 高斯模糊

对灰度图像进行模糊处理，使用简化的均值滤波实现。

**代码位置**：`Exercise05/`

**原理**：对每个像素，计算其周围窗口内所有像素的平均值

**特点**：

- 可调节模糊半径（默认 5，即 11×11 窗口）
- 自动处理边界像素
- **使用真实图像测试**：Grace Hopper 的经典照片
- 自动转换 RGB → 灰度 → 模糊
- 保存多个输出图像用于对比

**输出**：

- `output_original_gray.jpg`：转换后的灰度图
- `output_blurred_gpu.jpg`：GPU 模糊结果
- `output_blurred_cpu.jpg`：CPU 验证结果

**注意**：此实现使用均值滤波（简化版本），真正的高斯模糊需要使用高斯核函数加权。

#### 运行 Exercise05

```bash
cd Exercise05
make
make run
```

#### 预期输出

```text
==================================================
第三章 - 练习: 高斯模糊
==================================================

已加载图像: Grace_Hopper.jpg
图像尺寸: 600×600, 通道数: 3
模糊半径: 5 (窗口大小: 11×11)

步骤 1/3: RGB → 灰度
已保存: output_original_gray.jpg

步骤 2/3: GPU 模糊处理
处理完成，用时: 3.456 ms
已保存: output_blurred_gpu.jpg

步骤 3/3: CPU 验证
处理完成，用时: 189.234 ms
已保存: output_blurred_cpu.jpg

=== 结果验证 ===
✅ GPU 和 CPU 结果完全一致

加速比: 54.76x
✅ 测试完成！
```

**输出文件**：`output_original_gray.jpg`, `output_blurred_gpu.jpg`, `output_blurred_cpu.jpg`

---

## 📖 练习题解答

### 练习 1

**题目：** 在本章中，我们实现了一个矩阵乘法 kernel，其中每个线程生成一个输出矩阵元素。在这个练习中，你将实现不同的矩阵乘法 kernel 并比较它们。

**a. 编写一个 kernel，让每个线程生成一个输出矩阵行。填写该设计的执行配置参数。**

**解答：**

```cuda
__global__
void matrixMulRowKernel(float* M, float* N, float* P, int size){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size){
        // 处理该行的每个元素
        for (int col=0; col<size; ++col){
            float sum = 0;
            for (int j=0; j<size; ++j){
                sum += M[row * size + j] * N[j * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}
```

**执行配置**：

```cuda
dim3 dimBlock(256, 1, 1);  // 每个块 256 个线程
dim3 dimGrid((size + 255) / 256, 1, 1);  // 足够的块来覆盖所有行
matrixMulRowKernel<<<dimGrid, dimBlock>>>(M, N, P, size);
```

---

**b. 编写一个 kernel，让每个线程生成一个输出矩阵列。填写该设计的执行配置参数。**

**解答：**

```cuda
__global__
void matrixMulColKernel(float* M, float* N, float* P, int size){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size){
        // 处理该列的每个元素
        for (int row = 0; row < size; ++row){
            float sum = 0;
            for (int j = 0; j < size; ++j){
                sum += M[row * size + j] * N[j * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}
```

**执行配置**：

```cuda
dim3 dimBlock(256, 1, 1);  // 每个块 256 个线程
dim3 dimGrid((size + 255) / 256, 1, 1);  // 足够的块来覆盖所有列
matrixMulColKernel<<<dimGrid, dimBlock>>>(M, N, P, size);
```

---

**c. 分析这两种 kernel 设计的优缺点。**

**解答：**

**性能分析**：

1. **并行度**：两种方法都大大降低了并行度。对于 N×N 矩阵：
   - 元素级方法：N² 个并行线程
   - 行级/列级方法：只有 N 个并行线程

2. **工作负载分布**：
   - 行级方法：每个线程计算 N² 次乘加运算
   - 列级方法：每个线程计算 N² 次乘加运算
   - 元素级方法：每个线程只计算 N 次乘加运算

3. **非方阵的影响**：
   - **行级方法**：如果列数远大于行数（宽矩阵），每个线程工作量过大，效率低
   - **列级方法**：如果行数远大于列数（高矩阵），每个线程工作量过大，效率低

4. **内存访问模式**：
   - 行级方法在读取 N 矩阵时会有非连续访问
   - 列级方法在读取 M 矩阵时会有非连续访问

**结论**：这两种方法都不如元素级方法高效，无法充分利用 GPU 的大规模并行能力。

---

### 练习 2

**题目：** 矩阵向量乘法。编写一个矩阵向量乘法 kernel 和主机存根函数。

**解答：**

```cuda
__global__
void matrixVecMulKernel(float* B, float* c, float* result, int vector_size, int matrix_rows){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < matrix_rows){
        float sum = 0;
        for (int j=0; j < vector_size; ++j){
            sum += B[i * vector_size + j] * c[j];
        }
        result[i] = sum;
    }
}
```

**主机函数**：

```cuda
void matrixVecMul(float* h_B, float* h_c, float* h_result, int vector_size, int matrix_rows){
    float *d_B, *d_c, *d_result;
    
    // 分配设备内存
    cudaMalloc(&d_B, matrix_rows * vector_size * sizeof(float));
    cudaMalloc(&d_c, vector_size * sizeof(float));
    cudaMalloc(&d_result, matrix_rows * sizeof(float));
    
    // 复制数据到设备
    cudaMemcpy(d_B, h_B, matrix_rows * vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, vector_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 启动 kernel
    int blockSize = 256;
    int gridSize = (matrix_rows + blockSize - 1) / blockSize;
    matrixVecMulKernel<<<gridSize, blockSize>>>(d_B, d_c, d_result, vector_size, matrix_rows);
    
    // 复制结果回主机
    cudaMemcpy(h_result, d_result, matrix_rows * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_B);
    cudaFree(d_c);
    cudaFree(d_result);
}
```

---

### 练习 3

考虑以下 CUDA kernel 和调用它的相应主机函数：

```cuda
01 __global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
02     unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
03     unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
04     if (row < M && col < N) {
05         b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
06     }
07 }
08 void foo(float* a_d, float* b_d) {
09     unsigned int M = 150;
10     unsigned int N = 300;
11     dim3 bd(16, 32);
12     dim3 gd((N - 1) / 16 + 1, ((M - 1) / 32 + 1));
13     foo_kernel <<<gd, bd>>> (a_d, b_d, M, N);
14 }
```

**a. 每个块有多少个线程？**

**解答：** `16 × 32 = 512` 个线程

块维度由 `bd(16, 32)` 定义，所以每个块有 `16 × 32 = 512` 个线程。

---

**b. 网格中有多少个线程？**

**解答：** `48,640` 个线程

**计算过程**：

- 块数（见 c）：`19 × 5 = 95` 个块
- 总线程数：`95 × 512 = 48,640` 个线程

---

**c. 网格中有多少个块？**

**解答：** `19 × 5 = 95` 个块

**计算过程**：

网格维度 `gd = ((N - 1) / 16 + 1, (M - 1) / 32 + 1)`

- x 方向：`(300 - 1) / 16 + 1 = 299 / 16 + 1 = 18 + 1 = 19`
- y 方向：`(150 - 1) / 32 + 1 = 149 / 32 + 1 = 4 + 1 = 5`
- 总块数：`19 × 5 = 95`

---

**d. 有多少个线程执行第 05 行的代码？**

**解答：** `45,000` 个线程

**计算过程**：

需要确定有多少线程满足条件 `row < M && col < N`：

- `M = 150`，实际矩阵有 150 行
- `N = 300`，实际矩阵有 300 列
- 线程网格覆盖：
  - row 范围：`0` 到 `5 × 32 - 1 = 159`（但只有 `0-149` 满足 `row < 150`）
  - col 范围：`0` 到 `19 × 16 - 1 = 303`（但只有 `0-299` 满足 `col < 300`）

实际执行的线程：`150 × 300 = 45,000` 个

---

### 练习 4

**题目：** 考虑一个宽度为 400、高度为 500 的 2D 矩阵。矩阵存储为一维数组。指定第 20 行第 10 列的矩阵元素的数组索引：

**a. 如果矩阵以行优先顺序存储。**

**解答：** `8,010`

**公式**：`index = row × width + col`

**计算**：`20 × 400 + 10 = 8,010`

---

**b. 如果矩阵以列优先顺序存储。**

**解答：** `5,020`

**公式**：`index = col × height + row`

**计算**：`10 × 500 + 20 = 5,020`

---

### 练习 5

**题目：** 考虑一个宽度为 400、高度为 500、深度为 300 的 3D 张量。张量以行优先顺序存储为一维数组。指定 x=10、y=20、z=5 位置的张量元素的数组索引。

**解答：** `1,008,010`

**3D 张量行优先公式**：`index = z × (width × height) + y × width + x`

**计算**：

```text
index = 5 × (400 × 500) + 20 × 400 + 10
      = 5 × 200,000 + 8,000 + 10
      = 1,000,000 + 8,000 + 10
      = 1,008,010
```

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0 或更高版本
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（计算能力 3.5+）
- **Python**: 3.11+（可选，用于 Gradio 可视化）

## 📚 参考资料

本章内容主要参考：

- 别人仓库的 chapter-03 实现
- PMPP 第四版教材第三章

---

## 💡 学习建议

1. **理解多维索引**：掌握 2D/3D 线程到数据的映射
2. **动手实践**：参考别人仓库的代码实现
3. **图像处理应用**：运行 Gradio 可视化界面，理解高斯模糊效果
4. **性能对比**：比较不同矩阵乘法策略的性能差异
5. **内存布局**：理解行优先和列优先存储的区别

## 🚀 下一步

完成本章学习后，继续学习：

- 第四章：计算架构和调度
- 第五章：内存架构和数据局部性
- 第六章：性能考虑

---

**学习愉快！** 🎓
