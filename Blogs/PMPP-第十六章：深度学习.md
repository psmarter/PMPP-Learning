---
title: PMPP-第十六章：深度学习
date: 2026-01-19 23:07:59
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 深度学习
  - 卷积神经网络
  - cuDNN
categories: 知识分享
cover: /img/PMPP.jpg
---

## 前言

前面的章节学习了各种并行算法原语：归约、扫描、排序、稀疏矩阵、图遍历。这些技术在深度学习中都有应用。第十六章将这些技术串联起来，展示 GPU 如何加速**深度学习（Deep Learning）**——当今 GPU 最重要的应用领域之一。本章不深入讲解神经网络理论，而是聚焦于**计算视角**：神经网络的核心操作是什么？GPU 如何高效实现这些操作？

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 深度学习基础

### 神经网络的计算本质

无论是全连接层、卷积层还是注意力层，神经网络的核心操作都是：

1. **线性变换**：矩阵乘法（GEMM）
2. **非线性激活**：ReLU、Sigmoid、Softmax 等
3. **归约操作**：Pooling、Normalization

**计算量分布**（以 ResNet-50 为例）：

- 卷积层：~95% 的计算量
- 全连接层：~4%
- 其他：~1%

卷积才是 GPU 优化的主战场。

### 深度学习为什么需要 GPU

| 特性           | CPU       | GPU        |
| -------------- | --------- | ---------- |
| 核心数         | 8-64      | 数千       |
| 时钟频率       | 3-5 GHz   | 1-2 GHz    |
| 峰值算力(FP32) | ~1 TFLOPS | ~30 TFLOPS |
| 内存带宽       | ~100 GB/s | ~1000 GB/s |

深度学习的计算是**高度并行**的：

- 批量处理（Batch）：独立样本并行
- 空间并行：图像不同位置并行
- 通道并行：不同特征图并行

这是 GPU 的理想场景。

## 卷积层的 GPU 实现

### 卷积的计算复杂度

给定输入张量 `[N, C_in, H, W]` 和卷积核 `[C_out, C_in, K, K]`：

$$
\text{FLOPs} = 2 \times N \times C_{out} \times H_{out} \times W_{out} \times C_{in} \times K \times K
$$

对于 ResNet-50 的第一个卷积层：

- 输入：`[1, 3, 224, 224]`
- 卷积核：`[64, 3, 7, 7]`
- FLOPs ≈ 1.2 亿次

整个网络约需 40 亿次浮点操作。

### 直接卷积

最直观的实现，第七章已详细讲过：

```cuda
__global__ void conv2d_direct(float *input, float *kernel, float *output,
                               int N, int C_in, int H, int W, 
                               int C_out, int K) {
    int n = blockIdx.z;
    int c_out = blockIdx.y;
    int h_out = blockIdx.x * blockDim.x + threadIdx.x;
    int w_out = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (h_out < H_out && w_out < W_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; c_in++) {
            for (int kh = 0; kh < K; kh++) {
                for (int kw = 0; kw < K; kw++) {
                    int h_in = h_out + kh - K/2;
                    int w_in = w_out + kw - K/2;
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        sum += input[...] * kernel[...];
                    }
                }
            }
        }
        output[...] = sum;
    }
}
```

**问题**：循环太多，内存访问不规则，效率低。

### Im2col + GEMM

**核心思想**：把卷积转化为矩阵乘法。

**Im2col**：把输入的每个滑动窗口展开成一列。

```
输入图像 [C_in, H, W]，卷积核 [K, K]

Im2col 后：
  列数 = H_out × W_out
  行数 = C_in × K × K

每一列对应一个滑动窗口的展开
```

**转换后**：

```
输出 = Kernel × Im2col(Input)
[C_out, H_out×W_out] = [C_out, C_in×K×K] × [C_in×K×K, H_out×W_out]
```

这是标准的 GEMM！可以调用高度优化的 cuBLAS。

**代价**：Im2col 需要额外内存，约为原输入的 K² 倍。

### Winograd 卷积

**核心思想**：用更多加法换更少乘法。

对于 3×3 卷积，Winograd F(2×2, 3×3) 可以把乘法次数从 36 减少到 16。

**公式**：

$$
Y = A^T \left[ (G \cdot g \cdot G^T) \odot (B^T \cdot d \cdot B) \right] A
$$

其中：

- $g$：卷积核
- $d$：输入 Tile
- $G, B, A$：变换矩阵
- $\odot$：逐元素乘法

**优势**：计算量减少 2.25 倍（理论上）。

**限制**：

- 只适合小卷积核（3×3 最常用）
- 数值稳定性问题
- 实现复杂

### FFT 卷积

**核心思想**：卷积定理——时域卷积 = 频域逐元素乘法。

```
Output = IFFT(FFT(Input) ⊙ FFT(Kernel))
```

**复杂度**：O(N² log N) vs 直接卷积的 O(N² K²)

**适用场景**：大卷积核（K > 7）。深度学习中卷积核通常很小（1×1, 3×3），所以 FFT 卷积较少使用。

### cuDNN 策略选择

cuDNN 内置多种卷积算法，会根据问题规模自动选择：

| 算法                  | 适用场景 |
| --------------------- | -------- |
| IMPLICIT_GEMM         | 通用     |
| IMPLICIT_PRECOMP_GEMM | 大 Batch |
| GEMM                  | 小卷积核 |
| WINOGRAD              | 3×3 卷积 |
| FFT                   | 大卷积核 |

```cuda
cudnnConvolutionFwdAlgoPerf_t perfResults[8];
cudnnFindConvolutionForwardAlgorithm(
    handle, inputDesc, filterDesc, convDesc, outputDesc,
    8, &returnedAlgoCount, perfResults);

// 选择最快的算法
cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
```

## 全连接层

### 本质：矩阵乘法

```
y = W × x + b
[out_features] = [out_features, in_features] × [in_features] + [out_features]
```

批量处理：

```
Y = X × W^T + B
[batch, out] = [batch, in] × [in, out] + [batch, out]
```

直接调用 cuBLAS GEMM 即可。

### 优化：Tensor Core

从 Volta 架构开始，GPU 有专门的 Tensor Core：

| 操作           | CUDA Core | Tensor Core |
| -------------- | --------- | ----------- |
| 4×4 矩阵乘加   | 128 FLOPs | 1 周期      |
| 精度           | FP32      | FP16/BF16   |
| 峰值算力(A100) | 19 TFLOPS | 312 TFLOPS  |

使用 Tensor Core 需要：

1. 数据类型为 FP16 或 BF16
2. 矩阵维度是 8 或 16 的倍数
3. 使用 cuBLAS 或 WMMA API

## 激活函数

### ReLU

```cuda
__global__ void relu(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = max(x[i], 0.0f);
    }
}
```

计算简单，属于 **Memory-Bound**：访存时间远大于计算时间。

**Fusion 优化**：把 ReLU 融合到卷积 Kernel 中，避免额外的内存读写。

```cuda
// 卷积结果直接应用 ReLU
output[idx] = max(conv_result, 0.0f);
```

### Softmax

```cuda
// 数值稳定版本
__global__ void softmax(float *x, float *y, int n) {
    // 1. 找最大值（规约）
    float max_val = reduce_max(x, n);
    
    // 2. exp(x - max) 并求和（规约）
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    sum = reduce_sum(sum);
    
    // 3. 归一化
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}
```

需要两次归约，三次遍历数据。

## Pooling 层

### Max Pooling

```cuda
__global__ void max_pool2d(float *input, float *output,
                            int H, int W, int pool_size, int stride) {
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    
    float max_val = -FLT_MAX;
    for (int ph = 0; ph < pool_size; ph++) {
        for (int pw = 0; pw < pool_size; pw++) {
            int h_in = h_out * stride + ph;
            int w_in = w_out * stride + pw;
            max_val = max(max_val, input[h_in * W + w_in]);
        }
    }
    output[h_out * W_out + w_out] = max_val;
}
```

类似于卷积，但用 max 替代加权和。

### Global Average Pooling

对整个特征图求平均：

```cuda
// 就是一个 2D 归约！
float sum = reduce_2d(feature_map, H, W);
output = sum / (H * W);
```

第十章的归约技术直接适用。

## Batch Normalization

### 计算过程

1. **计算均值**：$\mu = \frac{1}{m} \sum x_i$（归约）
2. **计算方差**：$\sigma^2 = \frac{1}{m} \sum (x_i - \mu)^2$（归约）
3. **归一化**：$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$（逐元素）
4. **缩放平移**：$y = \gamma \hat{x} + \beta$（逐元素）

### GPU 实现要点

```cuda
__global__ void batch_norm(float *x, float *y, 
                           float *gamma, float *beta,
                           int N, int C, int H, int W) {
    int c = blockIdx.x;  // 每个 Block 处理一个通道
    
    // 1. 计算该通道的均值和方差
    float mean = compute_mean(x, c, N, H, W);
    float var = compute_var(x, c, mean, N, H, W);
    
    __syncthreads();
    
    // 2. 归一化并缩放
    for (int i = threadIdx.x; i < N * H * W; i += blockDim.x) {
        float val = x[index(i, c)];
        y[index(i, c)] = gamma[c] * (val - mean) / sqrtf(var + 1e-5f) + beta[c];
    }
}
```

关键是**按通道归约**，每个通道独立处理。

## 反向传播

### 计算图与自动微分

深度学习框架（PyTorch、TensorFlow）通过**计算图**追踪操作，自动计算梯度。

**前向传播**：

```
x → Conv → ReLU → Pool → FC → Softmax → Loss
```

**反向传播**：

```
dLoss/dL ← dL/dFC ← dFC/dPool ← dPool/dReLU ← dReLU/dConv ← dConv/dx
```

每个操作都有对应的梯度计算 Kernel。

### 卷积的反向传播

卷积的梯度计算也是卷积！

```
# 对输入的梯度
dL/dInput = Convolution(dL/dOutput, Kernel^T)

# 对权重的梯度
dL/dKernel = Convolution(Input, dL/dOutput)
```

所以卷积的优化同样适用于反向传播。

## cuDNN 接口

### 基本使用

```cuda
#include <cudnn.h>

// 1. 创建句柄
cudnnHandle_t handle;
cudnnCreate(&handle);

// 2. 创建张量描述符
cudnnTensorDescriptor_t inputDesc, outputDesc;
cudnnCreateTensorDescriptor(&inputDesc);
cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                           N, C, H, W);

// 3. 创建卷积描述符
cudnnConvolutionDescriptor_t convDesc;
cudnnCreateConvolutionDescriptor(&convDesc);
cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1,
                                 CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

// 4. 查询工作空间大小
size_t workspaceSize;
cudnnGetConvolutionForwardWorkspaceSize(handle, inputDesc, filterDesc,
                                         convDesc, outputDesc, algo, &workspaceSize);

// 5. 分配工作空间并执行
void *workspace;
cudaMalloc(&workspace, workspaceSize);

float alpha = 1.0f, beta = 0.0f;
cudnnConvolutionForward(handle, &alpha, inputDesc, input,
                        filterDesc, filter, convDesc, algo,
                        workspace, workspaceSize,
                        &beta, outputDesc, output);
```

### 常用操作

| 操作             | 函数                            |
| ---------------- | ------------------------------- |
| 卷积前向         | cudnnConvolutionForward         |
| 卷积反向（数据） | cudnnConvolutionBackwardData    |
| 卷积反向（权重） | cudnnConvolutionBackwardFilter  |
| 池化             | cudnnPoolingForward/Backward    |
| 激活             | cudnnActivationForward/Backward |
| BatchNorm        | cudnnBatchNormForward/Backward  |
| Softmax          | cudnnSoftmaxForward/Backward    |

## 混合精度训练

### FP16 的优势

| 精度 | 存储   | 带宽 | 算力(A100) |
| ---- | ------ | ---- | ---------- |
| FP32 | 4 字节 | 1×   | 19 TFLOPS  |
| FP16 | 2 字节 | 2×   | 312 TFLOPS |

### 自动混合精度（AMP）

```python
# PyTorch 示例
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # 自动转换到 FP16
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()  # 梯度缩放
    scaler.step(optimizer)
    scaler.update()
```

**Loss Scaling**：防止 FP16 梯度下溢，先放大 Loss，计算完梯度后缩小。

## 小结

第十六章展示了深度学习与 GPU 并行计算的紧密联系：

**核心操作**：深度学习的计算本质是矩阵乘法（GEMM）和卷积。这两个操作占据了 95% 以上的计算量。

**卷积实现**：直接卷积简单但低效；Im2col + GEMM 转化为矩阵乘法；Winograd 减少乘法次数；cuDNN 自动选择最优策略。

**融合优化**：把多个操作融合到一个 Kernel 中，减少内存访问。ReLU、BatchNorm 等常与卷积融合。

**Tensor Core**：专用硬件加速矩阵运算，FP16 峰值算力是 FP32 的 16 倍。混合精度训练已成为标准做法。

**cuDNN**：封装了所有优化，是深度学习框架的底层依赖。理解其原理有助于调优和调试。

深度学习是 GPU 计算的"杀手级应用"，正是这一需求推动了 GPU 硬件和软件的飞速发展。掌握本章内容，你就能理解 PyTorch、TensorFlow 在底层是如何工作的。

## 🚀 下一步

- 深入学习 cuDNN 的 API，尝试手动调用不同的卷积算法并对比性能
- 实现一个简单的卷积神经网络，从零开始构建前向和反向传播
- 探索混合精度训练，了解 FP16/BF16 的使用场景和注意事项
- 学习 Tensor Core 编程，使用 WMMA API 实现自定义的矩阵乘法
- 研究算子融合技术，将多个操作合并到一个 Kernel 中

---

## 📚 参考资料

- PMPP 第四版 Chapter 16
- [第十六章：深度学习](https://smarter.xin/posts/pmmpp-chapter16-deep-learning/)
- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- [NVIDIA cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/)
- Lavin, A., & Gray, S. (2016). *Fast Algorithms for Convolutional Neural Networks*. CVPR.

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
