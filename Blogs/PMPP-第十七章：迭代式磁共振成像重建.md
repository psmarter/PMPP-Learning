---
title: PMPP-第十七章：迭代式磁共振成像重建
date: 2026-01-23 09:08:10
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - MRI
  - 医学影像
  - 迭代重建
categories: 知识分享
cover: /img/PMPP.jpg
---

## 前言

前面的章节主要讨论了通用的并行算法和深度学习应用。第十七章转向一个具体的应用领域——**医学影像（Medical Imaging）**，特别是**磁共振成像（MRI，Magnetic Resonance Imaging）**的图像重建。MRI 重建是计算密集型任务，涉及大量的傅里叶变换和迭代优化，是 GPU 加速的理想场景。本章将展示如何将前面学到的并行技术（FFT、矩阵运算、迭代求解）综合应用于实际问题。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## MRI 成像基础

### MRI 物理原理简述

MRI 利用人体中氢原子核的磁共振现象成像：

1. **激发**：射频脉冲激发氢原子核
2. **弛豫**：原子核恢复平衡时发出信号
3. **采集**：接收线圈采集信号
4. **编码**：梯度磁场对空间位置进行编码

采集到的信号位于**k 空间（k-space）**——图像的傅里叶变换域。

### 从 k 空间到图像

**基本公式**：

$$
s(k_x, k_y) = \int \int m(x, y) \cdot e^{-i2\pi(k_x x + k_y y)} dx dy
$$

其中：

- $s(k_x, k_y)$：k 空间信号
- $m(x, y)$：图像（磁化强度分布）

**重建**就是从 $s$ 恢复 $m$，即**逆傅里叶变换**。

### 笛卡尔采样 vs 非笛卡尔采样

**笛卡尔采样**：k 空间数据在规则网格上采集。

- 优点：直接用 FFT 重建
- 缺点：采集速度受限

**非笛卡尔采样**（如螺旋、径向）：

- 优点：采集更快，对运动不敏感
- 缺点：不能直接 FFT，需要特殊处理

## 直接重建与迭代重建

### 直接重建

对于笛卡尔采样的全采样数据：

```
图像 = IFFT(k空间数据)
```

简单快速，但有两个问题：

1. **欠采样**：为加速扫描，往往只采集部分 k 空间数据
2. **非笛卡尔轨迹**：数据不在网格上

### 欠采样的问题

根据奈奎斯特定理，欠采样会导致**混叠伪影（Aliasing）**。

直接 IFFT 重建欠采样数据会产生严重的图像失真。

### 迭代重建

**思路**：把重建建模为**优化问题**。

$$
\hat{m} = \arg\min_m \frac{1}{2}\|Am - s\|_2^2 + \lambda R(m)
$$

其中：

- $A$：前向模型（编码矩阵）
- $s$：采集的 k 空间数据
- $R(m)$：正则化项（先验约束）
- $\lambda$：正则化权重

**迭代求解**：通过梯度下降、共轭梯度等方法逐步逼近最优解。

## 非均匀傅里叶变换（NUFFT）

### 问题

非笛卡尔采样的数据不在规则网格上，不能直接用 FFT。

**非均匀离散傅里叶变换（NUDFT）**：

$$
s(k_j) = \sum_{i=1}^{N} m(x_i) \cdot e^{-i2\pi k_j x_i}
$$

直接计算复杂度 O(MN)，太慢。

### NUFFT 算法

**核心思想**：先插值到规则网格，再用 FFT。

**Type-1 NUFFT（非均匀到均匀）**：

1. 把非均匀点的值"散布"到规则网格（卷积/插值）
2. 对规则网格执行 FFT
3. 去卷积校正

**Type-2 NUFFT（均匀到非均匀）**：

1. 对规则网格执行 IFFT
2. 在非均匀点插值采样

### GPU 实现要点

```cuda
__global__ void gridding_kernel(
    cuComplex *non_uniform_data,  // 非均匀采样数据
    float2 *trajectory,            // 采样轨迹 (kx, ky)
    cuComplex *grid,               // 输出网格
    int num_samples,
    int grid_size,
    float kernel_width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;
    
    float kx = trajectory[idx].x;
    float ky = trajectory[idx].y;
    cuComplex val = non_uniform_data[idx];
    
    // 找到影响的网格点范围
    int x_start = max(0, (int)(kx - kernel_width));
    int x_end = min(grid_size - 1, (int)(kx + kernel_width));
    int y_start = max(0, (int)(ky - kernel_width));
    int y_end = min(grid_size - 1, (int)(ky + kernel_width));
    
    // 散布到网格
    for (int gx = x_start; gx <= x_end; gx++) {
        for (int gy = y_start; gy <= y_end; gy++) {
            float weight = kaiser_bessel(kx - gx, ky - gy, kernel_width);
            atomicAdd(&grid[gy * grid_size + gx].x, val.x * weight);
            atomicAdd(&grid[gy * grid_size + gx].y, val.y * weight);
        }
    }
}
```

**关键优化**：

- 使用 Kaiser-Bessel 窗函数插值
- 原子操作处理网格冲突
- 过采样网格减少误差

## 共轭梯度法（CG）

### 算法概述

共轭梯度法求解线性系统 $Ax = b$：

```
初始化：x_0, r_0 = b - A*x_0, p_0 = r_0

循环：
    alpha_k = (r_k·r_k) / (p_k·A*p_k)
    x_{k+1} = x_k + alpha_k * p_k
    r_{k+1} = r_k - alpha_k * A*p_k
    beta_k = (r_{k+1}·r_{k+1}) / (r_k·r_k)
    p_{k+1} = r_{k+1} + beta_k * p_k
```

### GPU 实现

CG 的主要操作：

1. **矩阵-向量乘**：A*p（使用 NUFFT）
2. **向量内积**：r·r（归约）
3. **向量加法**：x + alpha*p（逐元素）

```cuda
void cg_solve(
    cuComplex *x,      // 解（图像）
    cuComplex *b,      // 右端项（k空间数据）
    int max_iter,
    float tol) {
    
    // 分配中间变量
    cuComplex *r, *p, *Ap;
    
    // r = b - A*x
    apply_forward_model(x, Ap);
    vector_subtract(b, Ap, r, n);
    vector_copy(r, p, n);
    
    float rr = vector_dot(r, r, n);
    
    for (int k = 0; k < max_iter && rr > tol; k++) {
        // Ap = A * p
        apply_forward_model(p, Ap);
        
        // alpha = rr / (p·Ap)
        float pAp = vector_dot(p, Ap, n);
        float alpha = rr / pAp;
        
        // x = x + alpha * p
        vector_axpy(alpha, p, x, n);
        
        // r = r - alpha * Ap
        vector_axpy(-alpha, Ap, r, n);
        
        // beta = rr_new / rr_old
        float rr_new = vector_dot(r, r, n);
        float beta = rr_new / rr;
        
        // p = r + beta * p
        vector_xpay(r, beta, p, n);
        
        rr = rr_new;
    }
}
```

### NUFFT 作为前向模型

在 MRI 重建中，$A$ 不是显式存储的矩阵，而是通过 NUFFT 计算：

```cuda
void apply_forward_model(cuComplex *image, cuComplex *kspace) {
    // 1. 图像空间 → k空间（Type-2 NUFFT）
    nufft_type2(image, kspace, trajectory, num_samples, grid_size);
}

void apply_adjoint_model(cuComplex *kspace, cuComplex *image) {
    // 2. k空间 → 图像空间（Type-1 NUFFT，共轭转置）
    nufft_type1(kspace, image, trajectory, num_samples, grid_size);
}
```

共轭梯度中的 $A^H A$ 操作变成：NUFFT^H · NUFFT。

## 正则化

### 为什么需要正则化

欠采样导致问题**欠定**：方程数少于未知数。

需要额外约束来选择"好"的解。

### 常见正则化项

**Tikhonov 正则化（L2）**：

$$
R(m) = \|m\|_2^2
$$

促进解的平滑，抑制噪声。

**全变分（TV）**：

$$
R(m) = \sum_i \sqrt{(\nabla_x m_i)^2 + (\nabla_y m_i)^2}
$$

保留边缘的同时抑制噪声。

**小波稀疏**：

$$
R(m) = \|\Psi m\|_1
$$

利用图像在小波域的稀疏性。

### GPU 实现 TV 正则化

```cuda
__global__ void gradient_kernel(float *image, float *grad_x, float *grad_y,
                                  int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < W && y < H) {
        int idx = y * W + x;
        
        // 前向差分
        grad_x[idx] = (x < W - 1) ? image[idx + 1] - image[idx] : 0;
        grad_y[idx] = (y < H - 1) ? image[idx + W] - image[idx] : 0;
    }
}

__global__ void tv_proximal_kernel(float *grad_x, float *grad_y,
                                     float lambda, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < W && y < H) {
        int idx = y * W + x;
        
        float gx = grad_x[idx];
        float gy = grad_y[idx];
        float norm = sqrtf(gx * gx + gy * gy) + 1e-8f;
        
        // 软阈值
        float shrink = fmaxf(1.0f - lambda / norm, 0.0f);
        grad_x[idx] = gx * shrink;
        grad_y[idx] = gy * shrink;
    }
}
```

## 多线圈成像

### 并行成像

现代 MRI 使用**多接收线圈**，每个线圈有不同的灵敏度分布：

$$
s_c(k) = \int S_c(x) \cdot m(x) \cdot e^{-i2\pi k \cdot x} dx
$$

其中 $S_c$ 是第 $c$ 个线圈的灵敏度。

### SENSE 重建

**SENSE（Sensitivity Encoding）**：利用线圈灵敏度差异解混叠。

$$
\hat{m} = \arg\min_m \sum_c \|A S_c m - s_c\|_2^2
$$

### GPU 实现

多线圈增加了计算量，但也增加了并行度：

```cuda
void sense_cg_iteration(
    cuComplex *image,           // [H, W]
    cuComplex **coil_data,      // [num_coils][num_samples]
    cuComplex **sensitivity,    // [num_coils][H, W]
    int num_coils) {
    
    // 并行处理每个线圈
    for (int c = 0; c < num_coils; c++) {
        // 应用灵敏度
        elementwise_multiply(image, sensitivity[c], coil_image, H * W);
        
        // NUFFT
        nufft_forward(coil_image, coil_kspace, trajectory);
        
        // 累加
        vector_add(residual, coil_residual, residual, num_samples);
    }
}
```

线圈间**完全独立**，适合 GPU 并行。

## 压缩感知 MRI

### 基本原理

**压缩感知（Compressed Sensing）**：如果信号在某个域是稀疏的，可以用远少于奈奎斯特采样数的测量重建。

**MRI 中的稀疏性**：

- 图像在小波域稀疏
- 图像梯度稀疏（分段平滑）

### CS-MRI 优化问题

$$
\hat{m} = \arg\min_m \frac{1}{2}\|Am - s\|_2^2 + \lambda_1 \|\Psi m\|_1 + \lambda_2 TV(m)
$$

### ADMM 求解

**交替方向乘子法（ADMM）**将问题分解为子问题：

```
1. 数据保真度子问题（CG 求解）
2. 小波稀疏子问题（软阈值）
3. TV 子问题（TV 去噪）
4. 更新拉格朗日乘子
```

每个子问题都可以高效并行。

## 性能优化

### 内存管理

MRI 数据量大（3D、多线圈、多时相）：

```
4 线圈 × 256³ 复数 = 4 × 256³ × 8 = 512 MB
```

**策略**：

- 流水线处理时相
- 压缩存储低秩数据
- 使用统一内存自动管理

### 计算优化

| 操作  | 优化策略              |
| ----- | --------------------- |
| FFT   | 使用 cuFFT，批量处理  |
| NUFFT | 纹理缓存加速插值      |
| 规约  | Warp Shuffle 高效求和 |
| TV    | 共享内存差分模板      |

### 多 GPU

MRI 重建天然适合多 GPU：

- 线圈间并行
- 切片间并行
- 时相间并行

```cuda
#pragma omp parallel for
for (int c = 0; c < num_coils; c++) {
    int device = c % num_gpus;
    cudaSetDevice(device);
    process_coil(c);
}
```

## 实际应用

### 加速倍数

| 技术     | 加速倍数 |
| -------- | -------- |
| 并行成像 | 2-4×     |
| 压缩感知 | 4-8×     |
| 两者结合 | 8-16×    |

**临床意义**：扫描时间从 10 分钟缩短到 1 分钟，减少患者不适，提高通量。

### GPU 加速效果

| 平台         | 256³ 重建时间 |
| ------------ | ------------- |
| 单核 CPU     | ~300 秒       |
| 多核 CPU (8) | ~40 秒        |
| GPU (V100)   | ~2 秒         |

**加速比**：150× 以上。

## 小结

第十七章展示了 GPU 在医学影像领域的强大能力：

**MRI 重建本质**：从 k 空间数据恢复图像，涉及傅里叶变换和优化问题。

**NUFFT**：处理非笛卡尔采样的核心算法。GPU 的网格化和 FFT 实现比 CPU 快两个数量级。

**迭代重建**：共轭梯度 + 正则化解决欠采样问题。每次迭代包含 NUFFT、规约、逐元素操作——都是 GPU 擅长的。

**多线圈 + 压缩感知**：进一步加速采集，但计算量增加。GPU 的并行能力正好应对。

**综合应用**：本章综合了前面学的 FFT、矩阵运算、规约、迭代求解等技术，展示了实际问题中如何组合使用这些工具。

MRI 重建只是医学影像的冰山一角。CT 重建、PET 重建、超声成像都有类似的计算密集型任务，GPU 正在改变整个医学影像领域。

## 🚀 下一步

- 实现一个简单的 NUFFT，从 Type-1 和 Type-2 开始，理解网格化和插值的过程
- 学习共轭梯度法的 GPU 实现，掌握迭代求解器的优化技巧
- 探索不同的正则化方法：TV、小波稀疏、字典学习
- 研究压缩感知 MRI，了解稀疏重建的理论基础
- 了解其他医学影像重建方法：CT 重建（FBP、迭代重建）、PET 重建

---

## 📚 参考资料

- PMPP 第四版 Chapter 17
- [第十七章：迭代式磁共振成像重建](https://smarter.xin/posts/pmmpp-chapter17-mri-reconstruction/)
- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- Lustig, M., Donoho, D., & Pauly, J. M. (2007). *Sparse MRI: The Application of Compressed Sensing for Rapid MR Imaging*. MRM.
- Fessler, J. A., & Sutton, B. P. (2003). *Nonuniform Fast Fourier Transforms Using Min-Max Interpolation*. IEEE TSP.

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
