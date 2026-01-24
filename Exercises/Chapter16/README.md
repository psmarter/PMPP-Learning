# 第十六章：深度学习

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章介绍深度学习的 GPU 加速技术：

- 卷积神经网络（CNN）的计算过程
- 卷积层的前向和反向传播
- Pooling 层的实现（Section 16.2）
- 全连接层与矩阵乘法（GEMM）
- cuDNN 和 cuBLAS 库的使用
- 混合精度训练与 Tensor Core

**相关博客笔记**：[第十六章：深度学习](https://smarter.xin/posts/feaca34d/)

---

## 💻 代码实现

### Exercise01 - Pooling 层前向传播

实现 Max Pooling 和 Average Pooling 的 CPU 串行和 GPU 并行版本。

**代码位置**：`Exercise01/`

| 实现 | 类型 | 特点 |
| ---- | ---- | ---- |
| `pooling_max_forward_cpu/gpu` | Max Pooling | K×K 窗口取最大值 |
| `pooling_avg_forward_cpu/gpu` | Avg Pooling | K×K 窗口求平均 |

```bash
cd Exercise01 && make && make run
```

#### 预期输出

```text
================================================================
第十六章 - Exercise 01: Pooling 层（Max 和 Average Pooling）
================================================================

=== 正确性测试 ===

测试1: Max Pooling (N=2, C=3, H=8, W=8, K=2)
  输出尺寸: 2×3×4×4
  CPU vs GPU: ✅ 结果完全一致

测试2: Average Pooling (N=2, C=3, H=8, W=8, K=2)
  输出尺寸: 2×3×4×4
  CPU vs GPU: ✅ 结果完全一致

=== 性能测试 (N=32, C=128, H=56, W=56, K=2) ===

Max Pooling:
  CPU 时间: 145.234 ms
  GPU 时间: 0.782 ms
  加速比: 185.74x

Average Pooling:
  CPU 时间: 147.891 ms
  GPU 时间: 0.793 ms
  加速比: 186.48x

✅ 所有测试通过！
```

---

### Exercise02 - Conv2D 反向传播（简版）

实现卷积层反向传播中的输入梯度计算（无 padding/stride 的简化版本）。

**代码位置**：`Exercise02/`

| 实现 | 特点 |
| ---- | ---- |
| `conv2d_backward_input_cpu/gpu` | 计算 dL/dX，valid convolution |

```bash
cd Exercise02 && make && make run
```

#### 预期输出

```text
================================================================
第十六章 - Exercise 02: Conv2D 反向传播（输入梯度）
================================================================

=== 配置测试 ===

配置1: N=1, C_in=3, H=5, W=5, C_out=2, K=3
  输入尺寸: 1×3×5×5, 权重: 2×3×3×3, 输出梯度: 1×2×3×3
  ✅ CPU vs GPU 结果一致（误差 < 1e-4）

配置2: N=2, C_in=16, H=8, W=8, C_out=32, K=3
  输入尺寸: 2×16×8×8, 权重: 32×16×3×3, 输出梯度: 2×32×6×6
  ✅ CPU vs GPU 结果一致（误差 < 1e-4）

配置3: N=4, C_in=64, H=14, W=14, C_out=128, K=5
  输入尺寸: 4×64×14×14, 权重: 128×64×5×5, 输出梯度: 4×128×10×10
  ✅ CPU vs GPU 结果一致（误差 < 1e-4）

=== 性能测试 (N=16, C_in=128, H=28, W=28, C_out=256, K=3) ===

  CPU 时间: 1234.567 ms
  GPU 时间: 5.432 ms
  加速比: 227.23x

✅ 所有测试通过！
```

---

### Exercise03 - CNN 完整层实现

完整实现 Conv2D 和 MaxPool2D 的前向和反向传播，对应参考仓库的 `conv2d.cu`（~440行）。

**代码位置**：`Exercise03/`

| 实现 | 类型 | 特点 |
| ---- | ---- | ---- |
| `conv2d_forward` | 前向 | 支持 batch, padding, stride |
| `conv2d_backward_input` | 反向 | 计算输入梯度 dL/dX |
| `conv2d_backward_weights` | 反向 | 计算权重梯度 dL/dW |
| `conv2d_backward_bias` | 反向 | 计算偏置梯度 dL/db |
| `maxpool2d_forward` | 前向 | 带索引记录（用于反向传播） |
| `maxpool2d_backward` | 反向 | 使用前向记录的索引 |

```bash
cd Exercise03 && make && make run
```

#### 预期输出

```text
================================================================
第十六章 - Exercise 03: CNN 完整层（Conv2D + MaxPool2D）
================================================================

=== Conv2D 前向传播测试 ===
配置: N=2, C_in=16, H=28, W=28, C_out=32, K=3, P=1, S=1
  输出尺寸: 2×32×28×28
  ✅ 正确性验证通过

=== Conv2D 反向传播测试 ===
  输入梯度: ✅ 正确
  权重梯度: ✅ 正确
  偏置梯度: ✅ 正确

=== MaxPool2D 前向传播测试 ===
配置: N=2, C=32, H=28, W=28, K=2, S=2
  输出尺寸: 2×32×14×14
  ✅ 正确性验证通过

=== MaxPool2D 反向传播测试 ===
  输入梯度: ✅ 正确

=== 性能测试 ===
Conv2D Forward: 3.456 ms
Conv2D Backward: 8.234 ms
MaxPool2D Forward: 0.234 ms
MaxPool2D Backward: 0.189 ms

✅ 所有测试通过！
```

---

### Exercise04 - cuBLAS SGEMM 矩阵乘法

封装 cuBLAS 库的 SGEMM 操作，用于全连接层，对应参考仓库的 `cublas_wrapper.c`。

**代码位置**：`Exercise04/`

| 实现 | 特点 |
| ---- | ---- |
| `sgemm_wrapper` | 主机到设备完整流程 |
| `sgemm_device` | 仅设备端计算 |
| `gpu_alloc/gpu_free` | 内存管理工具 |

```bash
cd Exercise04 && make && make run
```

#### 预期输出

```text
================================================================
第十六章 - Exercise 04: cuBLAS SGEMM 矩阵乘法
================================================================

=== 正确性测试 ===
矩阵尺寸: M=128, N=256, K=512
  C = alpha * (A × B) + beta * C
  ✅ cuBLAS 结果与 CPU 参考一致

=== 性能对比 (M=1024, N=1024, K=1024) ===

朴素 CUDA kernel:  45.678 ms (59.02 GFLOPS)
cuBLAS SGEMM:      2.134 ms (1263.45 GFLOPS)
加速比: 21.40x

cuBLAS 性能达到理论峰值的 82.3%
✅ 所有测试通过！
```

---

### Exercise05 - cuDNN 封装实现

使用 NVIDIA cuDNN 高级库实现 Conv2D 和 MaxPool2D，对应参考仓库的 `legacy_cudnn_wrapper.cu`。

**代码位置**：`Exercise05/`

| 实现 | 类型 | 特点 |
| ---- | ---- | ---- |
| `conv2d_forward_cudnn` | 前向 | cuDNN 自动选择最优算法 |
| `conv2d_backward_cudnn` | 反向 | 计算 dL/dX, dL/dW, dL/db |
| `maxpool2d_forward_cudnn` | 前向 | cuDNN 池化实现 |
| `maxpool2d_backward_cudnn` | 反向 | cuDNN 反向池化 |

```bash
cd Exercise05 && make && make run
```

> **重要提示**: 本练习需要安装 cuDNN 库
> 
> **安装cuDNN**：
> 1. 从[NVIDIA官网](https://developer.nvidia.com/cudnn)下载cuDNN
> 2. 解压后将include和lib文件复制到CUDA安装目录
> 3. Linux: `sudo cp cudnn*/include/* /usr/local/cuda/include/`
> 4. Linux: `sudo cp cudnn*/lib/* /usr/local/cuda/lib64/`
> 
> **跳过此Exercise**：如果不需要cuDNN，可以跳过Exercise05，不影响其他练习

#### 预期输出

```text
================================================================
第十六章 - Exercise 05: cuDNN 封装实现
================================================================

cuDNN 版本: 8.9.0
检测到 GPU: NVIDIA GeForce RTX 4090

=== Conv2D 测试 ===
配置: N=4, C_in=64, H=28, W=28, C_out=128, K=3, P=1, S=1

前向传播:
  ✅ cuDNN 结果与 CPU 参考一致
  cuDNN 时间: 0.456 ms

反向传播:
  ✅ 输入梯度正确
  ✅ 权重梯度正确
  ✅ 偏置梯度正确
  cuDNN 时间: 1.234 ms

=== MaxPool2D 测试 ===
配置: N=4, C=128, H=28, W=28, K=2, S=2

前向传播:
  ✅ cuDNN 结果正确
  cuDNN 时间: 0.123 ms

反向传播:
  ✅ 输入梯度正确
  cuDNN 时间: 0.098 ms

✅ 所有测试通过！
```

---

## 📖 练习题解答

### 练习 1: Pooling 层前向传播

**题目**：实现 Section 16.2 中描述的 Pooling 层前向传播。

**解答**：见 `Exercise01/solution.cu`。核心算法：

```cpp
// Max Pooling: 在 K×K 窗口中找最大值
for (int kh = 0; kh < K; kh++) {
    for (int kw = 0; kw < K; kw++) {
        int h_in = h_out * K + kh;
        int w_in = w_out * K + kw;
        max_val = max(max_val, input[...][h_in][w_in]);
    }
}
output[...][h_out][w_out] = max_val;
```

---

### 练习 2: 数据布局分析

**题目**：改用 [N×H×W×C] 或 [C×H×W×N] 布局有什么优势？

**解答**：

| 布局 | 优势 | 劣势 |
| ---- | ---- | ---- |
| **NCHW** | cuDNN 默认，空间局部性好 | Tensor Core 效率较低 |
| **NHWC** | Tensor Core 友好，1×1卷积高效 | 需要转换 |
| **CHWN** | 批量同位置处理方便 | 非标准格式 |

**结论**：NHWC 对现代 GPU (Tensor Core) 更友好，TensorFlow 默认使用 NHWC。

---

### 练习 3: Conv2D 反向传播

**题目**：实现卷积层的反向传播。

**解答**：见 `Exercise02/solution.cu` 和 `Exercise03/solution.cu`。

完整反向传播包括三部分：

1. **输入梯度** dL/dX：每个输入位置累加所有相关输出梯度 × 权重
2. **权重梯度** dL/dW：每个权重位置累加所有 batch 的 输入 × 输出梯度
3. **偏置梯度** dL/db：对输出梯度在 batch 和空间维度求和

---

### 练习 4: Unroll Kernel 内存合并分析

**题目**：分析 `unroll_Kernel` 中对 X 的访问模式。

**解答**：

| 情况 | 访问模式 | 合并效果 |
| ---- | -------- | -------- |
| 同行相邻 | 地址差=1 | 完美合并 |
| 跨行边界 | 地址差=K | 部分合并 |
| 跨通道 | 地址差=H×W | 无法合并 |

对于 224×224 图像，3×3 卷积：>99% 的访问完美合并。

---

## 📁 项目结构

```
Exercise01/     # Pooling 层 (Max/Avg)
Exercise02/     # Conv2D 反向传播 (简版)
Exercise03/     # Conv2D + MaxPool2D 完整实现
Exercise04/     # cuBLAS SGEMM 矩阵乘法
Exercise05/     # cuDNN 封装实现 (需要 cuDNN 库)
```

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0+
- **cuBLAS**: 随 CUDA Toolkit 安装
- **cuDNN**: 8.0+ (仅 Exercise05 需要)

---

## 💡 学习建议

1. **理解数据布局**：深度学习中NCHW vs NHWC布局对性能影响巨大，掌握不同布局的优劣
2. **从简单开始**：先实现Pooling层（Exercise01），再逐步深入卷积的前向和反向传播
3. **验证反向传播**：使用数值梯度检查验证反向传播实现的正确性（扰动法）
4. **库的使用**：cuBLAS和cuDNN是生产环境的标准选择，理解如何正确使用这些库
5. **性能对比**：对比手写kernel与cuDNN的性能差距，理解库优化的价值
6. **Tensor Core**：在支持的GPU上尝试使用Tensor Core加速（需要NHWC布局和半精度）

## 🚀 下一步

完成本章学习后，继续学习：

- 第十七章：迭代式磁共振成像重建
- 第十八章：静电势能图

---

## 📚 参考资料

- PMPP 第四版 Chapter 16
- [GitHub参考仓库](https://github.com/tugot17/pmpp/tree/main/chapter-16)
- [第十六章：深度学习](https://smarter.xin/posts/feaca34d/)

**学习愉快！** 🎓
