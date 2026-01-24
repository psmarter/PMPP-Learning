---
title: PMPP-第三章：多维网格和数据
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 矩阵乘法
  - 图像处理
categories: 知识分享
cover: /img/PMPP.jpg
abbrlink: 6b7045b6
date: 2026-01-12 15:19:16
---

## 前言

第二章的向量加法是一维数据，实际应用大多是多维的——图像是2D，矩阵是2D，深度学习张量是3D/4D。第三章讲解如何用CUDA的多维Grid/Block处理这些数据，核心案例是矩阵乘法和图像处理。虽然还是基础实现，但已经能看出GPU编程的思维和优化方向。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 多维线程组织

### 从一维到二维

第二章使用一维索引：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

处理矩阵若用一维，需要转换：

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int row = idx / width;  // 除法
int col = idx % width;  // 取模
```

除法和取模在 GPU 上有开销，可读性也差。使用二维更自然：

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### dim3类型

```cuda
dim3 blockDim(16, 16);     // 256 threads: 16×16
dim3 gridDim(64, 64);      // 4096 blocks
kernel<<<gridDim, blockDim>>>(args);
```

如果用整数会自动转为1D：

```cuda
kernel<<<256, 128>>>(args);  // 等价于 dim3(256,1,1), dim3(128,1,1)
```

### 配置原则

- **匹配数据维度**：图像用2D，体数据用3D
- **考虑内存访问**：同一warp的线程最好访问连续内存
- **硬件限制**：
  - 每block最多1024 threads
  - Grid的y/z维最多65535 blocks

### 常见错误

**1. 参数顺序**：`<<<grid, block>>>`不要反

**2. 向上取整**：

```cuda
// 错误：size整除时会多一个block
int blocks = size / threads + 1;

// 正确
int blocks = (size + threads - 1) / threads;
```

**3. 超过限制**：

```cuda
dim3 block(32, 32, 1);   // 32*32=1024，刚好
dim3 block(32, 32, 2);   // 2048，超限！
```

## 矩阵乘法

### 问题定义

P[i][j] = Σ(k=0→n-1) M[i][k] × N[k][j]

- M: m×n
- N: n×o  
- P: m×o

### 基础实现

每个线程计算一个输出元素：

```cuda
__global__ void matMul(float *M, float *N, float *P, 
                       int m, int n, int o) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < o) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += M[row * n + k] * N[k * o + col];
        }
        P[row * o + col] = sum;
    }
}
```

**索引公式**（行主序）：

- M[i][j] → `i * width + j`
- 第i行第j列在一维数组中的位置

**启动配置**：

```cuda
dim3 block(16, 16);
dim3 grid((o + 15) / 16, (m + 15) / 16);
matMul<<<grid, block>>>(d_M, d_N, d_P, m, n, o);
```

**边界检查**：`if (row < m && col < o)` 两个条件都要（grid向上取整会多线程）

### 性能瓶颈

**算术强度极低**：

```
每线程读取: 2n个float = 8n bytes
每线程计算: 2n FLOP
算术强度 = 2n / 8n = 0.25 FLOP/Byte
```

对于峰值10 TFLOPS、带宽500 GB/s的GPU：

```
达到峰值需要: 10T / 500G = 20 FLOP/Byte
实际只有: 0.25
利用率: 1.25%
```

GPU大部分时间在等数据。更严重的是数据重复读取：M[i][k]被第i行的所有线程读，总共被读m×o次，但每次都从全局内存读，没有复用。

**优化方向**（第4-5章）：用Shared Memory做Tiling，算术强度提升到10+ FLOP/Byte。

### 变体对比

**每线程一行**：

```cuda
int row = blockIdx.x * blockDim.x + threadIdx.x;
for (int col = 0; col < size; col++) {
    // 计算P[row][col]
}
```

**每线程一列**：类似，外层循环改为row

| 方案 | 并行度 | 适用场景 |
|------|--------|----------|
| 每线程一元素 | m×o (最高) | 通用 |
| 每线程一行 | m | m >> o |
| 每线程一列 | o | o >> m |

大多数情况下，每线程一元素的并行度最高。

## 矩阵-向量乘法

A[i] = Σ B[i][j] × C[j]

```cuda
__global__ void matVecMul(float *B, float *C, float *A, 
                          int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += B[row * n + j] * C[j];
        }
        A[row] = sum;
    }
}
```

只需一维，因为输出是向量。瓶颈是向量C被所有线程重复读取，可用Constant Memory或Shared Memory优化。

## 图像处理

### RGB转灰度

**公式**：Gray = 0.299R + 0.587G + 0.114B

**内存布局**（RGB交错）：

```
[R0 G0 B0][R1 G1 B1]...[R(w-1) G(w-1) B(w-1)]
[Rw Gw Bw]...
```

**Kernel**：

```cuda
__global__ void rgb2gray(unsigned char *rgb, unsigned char *gray,
                         int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        int rgbIdx = (row * width + col) * 3;
        int grayIdx = row * width + col;
        
        unsigned char r = rgb[rgbIdx];
        unsigned char g = rgb[rgbIdx + 1];
        unsigned char b = rgb[rgbIdx + 2];
        
        gray[grayIdx] = 0.299f*r + 0.587f*g + 0.114f*b;
    }
}
```

**性能**：带宽受限（读3字节，写1字节，计算很少），但访问连续，coalescing效果好。

### 高斯模糊

**模板计算**（Stencil）：每个像素由邻域加权求和

3×3高斯核：

```
1/16 * [1 2 1]
       [2 4 2]
       [1 2 1]
```

**Kernel**：

```cuda
__global__ void gaussianBlur(unsigned char *in, unsigned char *out,
                              int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界直接复制
    if (row == 0 || row == height-1 || 
        col == 0 || col == width-1) {
        if (row < height && col < width)
            out[row*width + col] = in[row*width + col];
        return;
    }
    
    if (row < height && col < width) {
        float sum = 0;
        // 3x3邻域
        sum += in[(row-1)*width + col-1] * 1.0f;
        sum += in[(row-1)*width + col  ] * 2.0f;
        sum += in[(row-1)*width + col+1] * 1.0f;
        sum += in[row*width + col-1] * 2.0f;
        sum += in[row*width + col  ] * 4.0f;
        sum += in[row*width + col+1] * 2.0f;
        sum += in[(row+1)*width + col-1] * 1.0f;
        sum += in[(row+1)*width + col  ] * 2.0f;
        sum += in[(row+1)*width + col+1] * 1.0f;
        
        out[row*width + col] = sum / 16.0f;
    }
}
```

**关键点**：
-边界处理：这里简化为复制，也可padding/镜像

- 数据重用：相邻像素的邻域重叠，但此版本没复用
- 优化：用Shared Memory让block协作加载tile+halo

## 内存布局

### 行主序（C/CUDA）

元素[row][col]的索引：

```cuda
index = row * width + col
```

例：4×3矩阵元素[2][1]：`2*3 + 1 = 7`

### 列主序（Fortran/MATLAB）

```cuda
index = col * height + row
```

同样元素[2][1]：`1*4 + 2 = 6`

### 三维张量

深度×高度×宽度，元素[z][y][x]：

```cuda
index = z * (height * width) + y * width + x
```

例：300×500×400 (D×H×W)，元素[5][20][10]：

```
5*(500*400) + 20*400 + 10 = 1,008,010
```

**注意**：深度学习框架维度顺序可能不同（PyTorch用NCHW，TensorFlow用NHWC）。

## 执行配置示例

书中习题3：

```cuda
dim3 bd(16, 32);                      // block: 16×32
dim3 gd((300-1)/16+1, (150-1)/32+1); // grid
// M=150, N=300
```

- 每block：16×32 = 512 threads
- Grid：(19, 5) = 95 blocks
- 总线程：95×512 = 48,640
- 有效线程（row<150 && col<300）：150×300 = 45,000
- 浪费：3,640 (7.5%)，可接受

## 性能分析思路

### Roofline模型

```
实际性能 = min(计算峰值, 带宽 × 算术强度)
```

要达到计算峰值需要：算术强度 ≥ 峰值FLOPS / 带宽

例如GPU 10 TFLOPS, 500 GB/s：需要 ≥ 20 FLOP/Byte

当前算法都远低于此，性能受内存限制。

### 提升算术强度

核心：**数据重用**

- **Tiling**：数据加载到Shared Memory重复使用
- **寄存器复用**：每线程计算多个元素
- **Kernel融合**：减少中间结果传输

矩阵乘法优化版能达到10+ FLOP/Byte，性能提升10倍以上。

## 小结

第三章从一维扩展到多维，核心是Grid/Block的灵活组织：

**多维索引**：`row = blockIdx.y*blockDim.y + threadIdx.y` 要熟练，二维、三维是自然扩展。

**内存布局**：行主序决定索引公式 `row*width + col`，搞错会访问错误数据。

**性能认知**：基础实现都是内存受限，算术强度0.1-0.25 FLOP/Byte，远低于需要的20。GPU计算能力强，但数据供应跟不上。

**优化方向**：提升算术强度 = 增加数据重用。这是第4-5章的重点。

**代码习惯**：

- 二维/三维都要严格边界检查
- 索引计算先row后col（行主序）
- 性能分析要量化（算术强度、带宽利用率）

理解了朴素实现的瓶颈，才能明白共享内存（Shared Memory）和分块（Tiling）的价值。第五章的优化技术会让同样的矩阵乘法性能提升10倍以上。

---

**参考资料：**

- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
