---
title: PMPP-第十八章：静电势能图
date: 2026-01-23 10:05:17
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 分子动力学
  - 静电势能
  - 科学计算
categories: 知识分享
cover: /img/PMPP.jpg
---

## 前言

第十七章探索了 MRI 重建这一医学影像应用。第十八章转向另一个重要的科学计算领域——**分子动力学（Molecular Dynamics）**中的静电势能计算。静电相互作用是分子模拟的核心，理解分子如何通过电荷相互作用，对药物设计、蛋白质折叠研究等具有重要意义。本章将展示如何利用 GPU 的并行能力高效计算**静电势能图（Electrostatic Potential Map）**。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 静电势能基础

### 库仑定律

两个点电荷之间的静电势能：

$$
V = \frac{k \cdot q}{r}
$$

其中：

- $k$：库仑常数
- $q$：电荷量
- $r$：距离

### 分子中的静电势能

一个分子由多个原子组成，每个原子携带部分电荷。空间中任意一点的静电势是所有原子贡献的总和：

$$
V(\mathbf{p}) = \sum_{i=1}^{N} \frac{k \cdot q_i}{|\mathbf{p} - \mathbf{r}_i|}
$$

其中：

- $\mathbf{p}$：空间中的网格点
- $\mathbf{r}_i$：第 $i$ 个原子的位置
- $q_i$：第 $i$ 个原子的部分电荷

### 静电势能图

**静电势能图**：在分子周围的 3D 网格上，计算每个网格点的静电势能值。

典型规模：

- 网格：256³ = 1670 万个点
- 原子：数千到数万个

**计算量**：网格点数 × 原子数 = 数千亿次运算

## 直接求和法

### 基本算法

最直接的方法——对每个网格点，遍历所有原子求和：

```c
void compute_potential_cpu(
    float *potential,     // 输出：[Gx, Gy, Gz]
    float *atoms,         // 原子坐标：[N, 3]
    float *charges,       // 原子电荷：[N]
    int N,                // 原子数
    int Gx, int Gy, int Gz,  // 网格尺寸
    float grid_spacing) {
    
    for (int gx = 0; gx < Gx; gx++) {
        for (int gy = 0; gy < Gy; gy++) {
            for (int gz = 0; gz < Gz; gz++) {
                float px = gx * grid_spacing;
                float py = gy * grid_spacing;
                float pz = gz * grid_spacing;
                
                float sum = 0.0f;
                for (int i = 0; i < N; i++) {
                    float dx = px - atoms[i * 3 + 0];
                    float dy = py - atoms[i * 3 + 1];
                    float dz = pz - atoms[i * 3 + 2];
                    float r = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (r > 0.001f) {  // 避免除以零
                        sum += charges[i] / r;
                    }
                }
                potential[gx * Gy * Gz + gy * Gz + gz] = sum;
            }
        }
    }
}
```

**复杂度**：O(G³ × N)，对于 256³ 网格和 10000 个原子 ≈ 10¹¹ 次操作。

### GPU 朴素实现

每个线程计算一个网格点：

```cuda
__global__ void compute_potential_naive(
    float *potential,
    float *atoms,     // [N, 3]
    float *charges,   // [N]
    int N,
    int Gx, int Gy, int Gz,
    float grid_spacing) {
    
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (gx >= Gx || gy >= Gy || gz >= Gz) return;
    
    float px = gx * grid_spacing;
    float py = gy * grid_spacing;
    float pz = gz * grid_spacing;
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float dx = px - atoms[i * 3 + 0];
        float dy = py - atoms[i * 3 + 1];
        float dz = pz - atoms[i * 3 + 2];
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        if (r > 0.001f) {
            sum += charges[i] / r;
        }
    }
    
    potential[gx * Gy * Gz + gy * Gz + gz] = sum;
}
```

**问题**：每个线程都要读取所有原子数据——大量重复的全局内存访问。

## 共享内存优化

### 原子数据分块

把原子数据分成小块，每块加载到共享内存，复用于 Block 内所有线程：

```cuda
#define TILE_SIZE 128

__global__ void compute_potential_tiled(
    float *potential,
    float *atoms,
    float *charges,
    int N,
    int Gx, int Gy, int Gz,
    float grid_spacing) {
    
    __shared__ float s_atoms[TILE_SIZE * 3];
    __shared__ float s_charges[TILE_SIZE];
    
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;
    
    float px = gx * grid_spacing;
    float py = gy * grid_spacing;
    float pz = gz * grid_spacing;
    
    float sum = 0.0f;
    
    // 分块处理原子
    for (int tile = 0; tile < N; tile += TILE_SIZE) {
        // 协作加载原子数据到共享内存
        int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        int num_threads = blockDim.x * blockDim.y * blockDim.z;
        
        for (int i = tid; i < TILE_SIZE && (tile + i) < N; i += num_threads) {
            int atom_idx = tile + i;
            s_atoms[i * 3 + 0] = atoms[atom_idx * 3 + 0];
            s_atoms[i * 3 + 1] = atoms[atom_idx * 3 + 1];
            s_atoms[i * 3 + 2] = atoms[atom_idx * 3 + 2];
            s_charges[i] = charges[atom_idx];
        }
        __syncthreads();
        
        // 计算这一块原子的贡献
        int tile_atoms = min(TILE_SIZE, N - tile);
        for (int i = 0; i < tile_atoms; i++) {
            float dx = px - s_atoms[i * 3 + 0];
            float dy = py - s_atoms[i * 3 + 1];
            float dz = pz - s_atoms[i * 3 + 2];
            float r = sqrtf(dx*dx + dy*dy + dz*dz);
            if (r > 0.001f) {
                sum += s_charges[i] / r;
            }
        }
        __syncthreads();
    }
    
    if (gx < Gx && gy < Gy && gz < Gz) {
        potential[gx * Gy * Gz + gy * Gz + gz] = sum;
    }
}
```

### 性能分析

| 版本  | 全局内存读取次数 | 加速比 |
| ----- | ---------------- | ------ |
| 朴素  | G³ × N           | 1×     |
| Tiled | G³ × N / B + N   | ~10×   |

其中 B 是 Block 内线程数。

## 常量内存优化

### 原子数据放入常量内存

如果原子数不太多（< 16K），可以放入常量内存：

```cuda
#define MAX_ATOMS 16000

__constant__ float c_atoms[MAX_ATOMS * 4];  // x, y, z, charge

void setup_atoms(float *atoms, float *charges, int N) {
    float atoms_packed[MAX_ATOMS * 4];
    for (int i = 0; i < N; i++) {
        atoms_packed[i * 4 + 0] = atoms[i * 3 + 0];
        atoms_packed[i * 4 + 1] = atoms[i * 3 + 1];
        atoms_packed[i * 4 + 2] = atoms[i * 3 + 2];
        atoms_packed[i * 4 + 3] = charges[i];
    }
    cudaMemcpyToSymbol(c_atoms, atoms_packed, N * 4 * sizeof(float));
}

__global__ void compute_potential_const(
    float *potential,
    int N,
    int Gx, int Gy, int Gz,
    float grid_spacing) {
    
    // ... 网格点坐标计算 ...
    
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float ax = c_atoms[i * 4 + 0];
        float ay = c_atoms[i * 4 + 1];
        float az = c_atoms[i * 4 + 2];
        float q  = c_atoms[i * 4 + 3];
        
        float dx = px - ax;
        float dy = py - ay;
        float dz = pz - az;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);
        if (r > 0.001f) {
            sum += q / r;
        }
    }
    
    // ... 写入结果 ...
}
```

**优势**：

- 常量内存有专用缓存
- 广播机制：Warp 内线程访问同一地址只需一次读取

## 截断方法

### 为什么截断

真实分子模拟中，远距离原子的贡献很小。可以设置**截断半径**，只计算距离小于截断半径的原子。

$$
V(\mathbf{p}) = \sum_{|\mathbf{p} - \mathbf{r}_i| < r_{cut}} \frac{k \cdot q_i}{|\mathbf{p} - \mathbf{r}_i|}
$$

### 空间分区

为了快速找到"附近"的原子，使用**空间哈希**或**Cell List**：

1. 把空间划分成小格子（边长 = 截断半径）
2. 每个原子分配到所在格子
3. 计算网格点时，只遍历邻近 27 个格子中的原子

```cuda
__global__ void compute_potential_cutoff(
    float *potential,
    int *cell_start,    // [Cx, Cy, Cz] 每个格子的原子起始索引
    int *cell_count,    // [Cx, Cy, Cz] 每个格子的原子数
    float *sorted_atoms,
    float *sorted_charges,
    float cutoff,
    int Gx, int Gy, int Gz,
    int Cx, int Cy, int Cz,
    float grid_spacing,
    float cell_size) {
    
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (gx >= Gx || gy >= Gy || gz >= Gz) return;
    
    float px = gx * grid_spacing;
    float py = gy * grid_spacing;
    float pz = gz * grid_spacing;
    
    // 确定所在格子
    int cx = (int)(px / cell_size);
    int cy = (int)(py / cell_size);
    int cz = (int)(pz / cell_size);
    
    float sum = 0.0f;
    
    // 遍历邻近 27 个格子
    for (int dcx = -1; dcx <= 1; dcx++) {
        for (int dcy = -1; dcy <= 1; dcy++) {
            for (int dcz = -1; dcz <= 1; dcz++) {
                int ncx = cx + dcx;
                int ncy = cy + dcy;
                int ncz = cz + dcz;
                
                if (ncx < 0 || ncx >= Cx || ncy < 0 || ncy >= Cy || ncz < 0 || ncz >= Cz)
                    continue;
                
                int cell_idx = ncx * Cy * Cz + ncy * Cz + ncz;
                int start = cell_start[cell_idx];
                int count = cell_count[cell_idx];
                
                for (int i = 0; i < count; i++) {
                    int atom_idx = start + i;
                    float dx = px - sorted_atoms[atom_idx * 3 + 0];
                    float dy = py - sorted_atoms[atom_idx * 3 + 1];
                    float dz = pz - sorted_atoms[atom_idx * 3 + 2];
                    float r = sqrtf(dx*dx + dy*dy + dz*dz);
                    
                    if (r > 0.001f && r < cutoff) {
                        sum += sorted_charges[atom_idx] / r;
                    }
                }
            }
        }
    }
    
    potential[gx * Gy * Gz + gy * Gz + gz] = sum;
}
```

### 复杂度分析

| 方法        | 复杂度         | 适用场景 |
| ----------- | -------------- | -------- |
| 直接求和    | O(G³ × N)      | 小系统   |
| 截断 + Cell | O(G³ × ρ × r³) | 大系统   |

其中 ρ 是原子密度，r 是截断半径。

## 多层网格方法

### Ewald 求和

对于周期性边界条件，需要考虑**所有周期镜像**的贡献。

**Ewald 方法**将求和分成两部分：

1. **实空间**：截断求和（短程）
2. **倒空间**：FFT 求和（长程）

### PME（Particle Mesh Ewald）

PME 是 Ewald 的高效变体：

1. 把电荷分配到网格
2. 对网格做 FFT
3. 在倒空间计算势能
4. 做 IFFT
5. 插值回原子位置

**复杂度**：O(N log N) vs 直接 Ewald 的 O(N^1.5)

### GPU 上的 PME

```cuda
void pme_potential(
    float *potential,
    float *atoms,
    float *charges,
    int N,
    int grid_size) {
    
    // 1. 电荷分配到网格（B-spline 插值）
    charge_spreading<<<...>>>(charges, atoms, grid_charges, N, grid_size);
    
    // 2. 3D FFT
    cufftExecC2C(plan, grid_charges, grid_kspace, CUFFT_FORWARD);
    
    // 3. 倒空间操作（乘以 Green 函数）
    reciprocal_kernel<<<...>>>(grid_kspace, grid_size);
    
    // 4. 3D IFFT
    cufftExecC2C(plan, grid_kspace, grid_potential, CUFFT_INVERSE);
    
    // 5. 插值回网格点
    interpolate_kernel<<<...>>>(grid_potential, potential, grid_size);
}
```

## 多 GPU 实现

### 数据并行

对于大网格，可以将网格分割到多个 GPU：

```cuda
void multi_gpu_potential(
    float *potential,
    float *atoms,
    float *charges,
    int N,
    int Gx, int Gy, int Gz,
    int num_gpus) {
    
    int slices_per_gpu = Gx / num_gpus;
    
    #pragma omp parallel for num_threads(num_gpus)
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(gpu);
        
        int gx_start = gpu * slices_per_gpu;
        int gx_end = (gpu == num_gpus - 1) ? Gx : (gpu + 1) * slices_per_gpu;
        
        compute_potential_kernel<<<...>>>(
            potential + gx_start * Gy * Gz,
            atoms_d[gpu], charges_d[gpu], N,
            gx_start, gx_end, Gy, Gz, grid_spacing);
    }
    
    // 收集结果
    // ...
}
```

### 注意事项

- 每个 GPU 需要完整的原子数据副本
- 边界区域可能需要重叠计算
- 使用 NCCL 高效通信

## 应用：分子对接

### 药物设计中的应用

静电势能图用于**分子对接**：预测配体分子与蛋白质靶点的结合方式。

**流程**：

1. 计算蛋白质的静电势能图
2. 配体在势能图中搜索最优位置
3. 评估结合亲和力

### 实时可视化

GPU 计算的静电势能图可以实时渲染：

```cuda
// 将势能图转换为颜色
__global__ void potential_to_color(
    float *potential,
    uchar4 *colors,
    float min_val, float max_val,
    int num_points) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    float val = potential[idx];
    float normalized = (val - min_val) / (max_val - min_val);
    
    // 红色表示正电势，蓝色表示负电势
    colors[idx].x = (unsigned char)(255 * fmaxf(normalized, 0.0f));
    colors[idx].y = 0;
    colors[idx].z = (unsigned char)(255 * fmaxf(-normalized, 0.0f));
    colors[idx].w = 255;
}
```

## 性能总结

### 各优化的加速效果

| 优化技术        | 相对加速 | 累计加速 |
| --------------- | -------- | -------- |
| GPU 朴素        | 50×      | 50×      |
| 共享内存 Tiling | 10×      | 500×     |
| 常量内存        | 1.5×     | 750×     |
| 截断 + Cell     | 5-20×    | 3000×+   |

### 实际性能

256³ 网格，10000 个原子：

- CPU（单核）：~600 秒
- GPU（V100）：~0.2 秒

**加速比**：3000× 以上

## 小结

第十八章展示了静电势能计算的 GPU 加速：

**问题本质**：N-body 类型的全对问题，每个网格点需要遍历所有原子。直接计算复杂度 O(G³ × N)。

**共享内存优化**：把原子数据分块加载到共享内存，Block 内线程共享，大幅减少全局内存访问。

**常量内存**：对于小规模原子数据，利用常量缓存和广播机制进一步加速。

**截断方法**：利用物理特性（远场贡献小）减少计算量。Cell List 数据结构快速查找邻近原子。

**PME 方法**：对于周期性系统，用 FFT 处理长程相互作用，复杂度降至 O(N log N)。

**多 GPU 扩展**：网格天然可分割，适合数据并行。

静电势能计算是分子动力学模拟的核心组件。掌握本章技术，就能理解 GROMACS、AMBER 等分子动力学软件的 GPU 加速原理。

## 🚀 下一步

- 实现一个完整的静电势能计算程序，从直接求和开始，逐步优化到共享内存和截断方法
- 学习 PME 方法，实现基于 FFT 的长程相互作用计算
- 探索其他 N-body 问题：引力模拟、流体动力学中的粒子方法
- 了解分子动力学模拟的完整流程：力场计算、积分器、温度/压力控制
- 研究 GROMACS 或 AMBER 的 GPU 加速实现，学习工业级优化技巧

---

## 📚 参考资料

- PMPP 第四版 Chapter 18
- [第十八章：静电势能图](https://smarter.xin/posts/pmmpp-chapter18-electrostatic-potential/)
- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- Stone, J. E., et al. (2007). *Accelerating Molecular Modeling Applications with GPU Computing*. JCC.
- Darden, T., et al. (1993). *Particle Mesh Ewald: An N·log(N) Method for Ewald Sums*. JCP.

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
