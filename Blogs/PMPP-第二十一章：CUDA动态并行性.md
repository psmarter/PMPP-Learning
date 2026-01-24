---
title: PMPP-第二十一章：CUDA动态并行性
date: 2026-01-24 16:20:44
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 动态并行
  - 递归算法
  - 四叉树
categories: 知识分享
cover: /img/PMPP.jpg
---

## 前言

第二十章介绍了多 GPU 集群编程。第二十一章回到单 GPU 的高级特性——**动态并行性（Dynamic Parallelism）**。传统 CUDA 程序中，只有 CPU 能启动核函数；而动态并行性允许**GPU 核函数直接启动子核函数**，无需返回 CPU。这一特性对递归算法、自适应计算、不规则数据结构（如树、图）的处理特别有用。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 什么是动态并行性

### 传统 CUDA 模型的限制

传统 CUDA 程序中：

```
CPU 代码 → 启动 kernel1 → GPU 执行 → 返回 CPU
         → 启动 kernel2 → GPU 执行 → 返回 CPU
         → ...
```

**问题**：

1. 每次启动 kernel 都需要 CPU 参与
2. 递归算法难以表达
3. 自适应算法需要多次 CPU-GPU 往返

### 动态并行性

**动态并行性**：允许 GPU kernel 直接启动子 kernel（child kernel）。

```
CPU 代码 → 启动父 kernel → GPU 执行
                           ├→ 启动子 kernel1 → GPU 执行
                           ├→ 启动子 kernel2 → GPU 执行
                           └→ ...
```

**优势**：

- 减少 CPU-GPU 通信
- 自然表达递归算法
- 根据数据特性动态调整计算

### 硬件要求

动态并行性需要计算能力 3.5 或更高的 GPU。

```bash
# 编译时需要特殊选项
nvcc -arch=sm_35 -rdc=true my_program.cu -lcudadevrt
```

- `-rdc=true`：启用可重定位设备代码
- `-lcudadevrt`：链接设备运行时库

## 基本语法

### 在设备代码中启动 kernel

```cuda
__global__ void child_kernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] *= 2;
    }
}

__global__ void parent_kernel(int *data, int n) {
    // 在设备代码中启动子 kernel
    child_kernel<<<1, 32>>>(data, n);
    
    // 等待子 kernel 完成
    cudaDeviceSynchronize();
}
```

### 设备端同步

```cuda
// 等待当前线程启动的所有子 kernel 完成
cudaDeviceSynchronize();

// 等待特定流中的操作完成
cudaStreamSynchronize(stream);
```

**注意**：父 kernel 退出时会隐式等待所有子 kernel 完成。

### 设备端内存管理

```cuda
__global__ void parent_kernel() {
    float *temp;
    
    // 在设备端分配内存
    cudaMalloc(&temp, 1024 * sizeof(float));
    
    // 使用内存...
    child_kernel<<<1, 256>>>(temp, 1024);
    cudaDeviceSynchronize();
    
    // 释放内存
    cudaFree(temp);
}
```

## 内存可见性

### 哪些内存子 kernel 可以访问

| 内存类型 | 子 kernel 可访问 | 说明             |
| -------- | ---------------- | ---------------- |
| 全局内存 | ✓                | 父子共享         |
| 常量内存 | ✓                | 编译时确定，共享 |
| 纹理内存 | ✓                | 父子共享         |
| 共享内存 | ✗                | 仅限当前 Block   |
| 局部内存 | ✗                | 仅限当前线程     |

### 内存一致性

父 kernel 的全局内存写入对子 kernel 可见，需要遵循一定规则：

```cuda
__global__ void parent_kernel(int *data) {
    // 写入全局内存
    data[threadIdx.x] = threadIdx.x * 2;
    
    // 确保写入完成（块内同步）
    __threadfence();
    
    // 启动子 kernel
    if (threadIdx.x == 0) {
        child_kernel<<<1, 32>>>(data);
    }
}
```

**关键点**：

- 父线程在启动子 kernel 前的写入对子 kernel 可见
- 子 kernel 的写入在 `cudaDeviceSynchronize()` 后对父线程可见

## 流与并发

### 默认流行为

```cuda
__global__ void parent_kernel() {
    // 这两个 kernel 在同一个 Block 的默认流中顺序执行
    child_kernel1<<<1, 32>>>();
    child_kernel2<<<1, 32>>>();  // 等待 child_kernel1 完成
}
```

**重要**：同一 Block 内的线程共享默认流，因此子 kernel 会串行执行。

### 使用独立流实现并发

```cuda
__global__ void parent_kernel() {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    
    // 在独立流中启动，可以与其他 kernel 并发
    child_kernel<<<1, 32, 0, stream>>>();
    
    cudaStreamDestroy(stream);
}
```

### 并发子 kernel 示例

```cuda
__global__ void parent_kernel(float *data, int n) {
    int chunk_size = n / 4;
    cudaStream_t streams[4];
    
    for (int i = 0; i < 4; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        child_kernel<<<1, 256, 0, streams[i]>>>(
            data + i * chunk_size, chunk_size);
    }
    
    // 等待所有流完成
    cudaDeviceSynchronize();
    
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
    }
}
```

## 启动池配置

### 什么是启动池

每次子 kernel 启动需要从**启动池**分配资源。

**两种池**：

1. **固定大小池**：默认 2048 个槽位，预分配
2. **虚拟化池**：动态扩展，但性能较低

### 配置启动池

```cuda
// 获取当前设备属性
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// 设置固定池大小
cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 8192);
```

**建议**：

- 如果预期子 kernel 数量超过默认值，增大固定池
- 如果子 kernel 数量可预测，设置为该值

## 应用示例：Bezier 曲线细分

### 问题描述

Bezier 曲线由控制点定义。曲线越弯曲，需要越多采样点才能平滑显示。

**自适应细分**：根据曲率决定采样点数量。

### 数据结构

```cuda
#define MAX_TESS_POINTS 32

struct BezierLine {
    float2 CP[3];                      // 3 个控制点
    float2 vertexPos[MAX_TESS_POINTS]; // 细分后的顶点
    int nVertices;                     // 顶点数量
};
```

### 计算曲率

```cuda
__device__ float computeCurvature(float2 *cp) {
    // 计算首尾连线长度
    float dx = cp[2].x - cp[0].x;
    float dy = cp[2].y - cp[0].y;
    float line_length = sqrtf(dx * dx + dy * dy);
    
    if (line_length < 0.001f) return 0.0f;
    
    // 计算中点到直线的距离（曲率近似）
    float cross = fabsf((cp[1].x - cp[0].x) * dy - 
                        (cp[1].y - cp[0].y) * dx);
    return cross / line_length;
}
```

### 子 kernel：计算细分点

```cuda
__global__ void computeBezierLine_child(
    int lidx, BezierLine *bLines, int nTessPoints) {
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < nTessPoints) {
        // 计算参数 u ∈ [0, 1]
        float u = (float)idx / (float)(nTessPoints - 1);
        float omu = 1.0f - u;
        
        // 二次 Bezier 基函数
        float B[3];
        B[0] = omu * omu;
        B[1] = 2.0f * u * omu;
        B[2] = u * u;
        
        // 计算点位置
        float2 pos = {0, 0};
        for (int i = 0; i < 3; i++) {
            pos.x += B[i] * bLines[lidx].CP[i].x;
            pos.y += B[i] * bLines[lidx].CP[i].y;
        }
        
        bLines[lidx].vertexPos[idx] = pos;
    }
}
```

### 父 kernel：动态决定细分程度

```cuda
__global__ void computeBezierLines_parent(
    BezierLine *bLines, int nLines) {
    
    int lidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lidx < nLines) {
        // 根据曲率计算需要的顶点数
        float curvature = computeCurvature(bLines[lidx].CP);
        int nVertices = min(max((int)(curvature * 16.0f), 4), 
                            MAX_TESS_POINTS);
        bLines[lidx].nVertices = nVertices;
        
        // 动态启动子 kernel
        int blocks = (nVertices + 31) / 32;
        computeBezierLine_child<<<blocks, 32>>>(
            lidx, bLines, nVertices);
    }
}
```

### 动态 vs 静态对比

```cuda
// 静态版本：每条曲线使用固定 Block，循环处理
__global__ void computeBezierLines_static(
    BezierLine *bLines, int nLines) {
    
    int bidx = blockIdx.x;
    if (bidx < nLines) {
        float curvature = computeCurvature(bLines[bidx].CP);
        int nVertices = min(max((int)(curvature * 16.0f), 4), 
                            MAX_TESS_POINTS);
        bLines[bidx].nVertices = nVertices;
        
        // 用循环代替子 kernel
        for (int inc = 0; inc < nVertices; inc += blockDim.x) {
            int idx = inc + threadIdx.x;
            if (idx < nVertices) {
                // ... 计算顶点 ...
            }
        }
    }
}
```

**对比**：

| 方面     | 动态并行           | 静态版本     |
| -------- | ------------------ | ------------ |
| 代码结构 | 更自然             | 需要手动循环 |
| 资源利用 | 子 kernel 精确配置 | 可能浪费线程 |
| 启动开销 | 较高               | 无额外开销   |
| 适用场景 | 工作量变化大       | 工作量均匀   |

## 应用示例：四叉树构建

### 问题描述

**四叉树（Quadtree）**：递归地将 2D 空间划分为四个象限，用于空间索引、碰撞检测、图像压缩等。

```
初始区域         第一次划分         继续划分...
┌─────────┐     ┌────┬────┐     ┌──┬──┬────┐
│         │     │ TL │ TR │     │  │  │ TR │
│    *    │  →  ├────┼────┤  →  ├──┼──┼────┤
│   **    │     │ BL │ BR │     │  │  │    │
└─────────┘     └────┴────┘     └──┴──┴────┘
```

### 递归算法

```
function build_quadtree(node, points):
    if depth > max_depth or len(points) < threshold:
        return  // 停止递归
    
    // 将点分类到四个象限
    for each point in points:
        classify to TL, TR, BL, or BR
    
    // 递归处理每个象限
    build_quadtree(node.TL, TL_points)
    build_quadtree(node.TR, TR_points)
    build_quadtree(node.BL, BL_points)
    build_quadtree(node.BR, BR_points)
```

### 数据结构

```cuda
class Quadtree_node {
    int m_id;                  // 节点 ID
    Bounding_box m_bbox;       // 边界框
    int m_begin, m_end;        // 点范围 [begin, end)
    
public:
    __device__ int num_points() const { return m_end - m_begin; }
    // ...
};

struct Parameters {
    int point_selector;         // 双缓冲选择器
    int num_nodes_at_this_level; // 当前层节点数
    int depth;                  // 当前深度
    int max_depth;              // 最大深度
    int min_points_per_node;    // 停止阈值
};
```

### 核心 kernel

```cuda
__global__ void build_quadtree_kernel(
    Quadtree_node *nodes, Points *points, Parameters params) {
    
    __shared__ int smem[8];  // 每个象限的点数和偏移
    
    Quadtree_node *node = &nodes[blockIdx.x];
    int num_points = node->num_points();
    
    // 检查停止条件
    if (params.depth >= params.max_depth || 
        num_points <= params.min_points_per_node) {
        return;
    }
    
    // 计算边界框中心
    float2 center;
    node->bounding_box().compute_center(&center);
    
    // 统计每个象限的点数
    count_points_in_children(points[params.point_selector], 
                              smem, node, center);
    
    // 计算重排偏移
    scan_for_offsets(node->points_begin(), smem);
    
    // 重排点到双缓冲
    reorder_points(points[(params.point_selector + 1) % 2],
                   points[params.point_selector], smem, node, center);
    
    // 递归启动子 kernel
    if (threadIdx.x == 0) {
        Quadtree_node *children = 
            &nodes[params.num_nodes_at_this_level + blockIdx.x * 4];
        
        prepare_children(children, node, smem, center);
        
        Parameters next_params(params, true);  // 更新参数
        
        for (int i = 0; i < 4; i++) {
            if (smem[i] > 0) {  // 只处理非空象限
                build_quadtree_kernel<<<1, 32>>>(
                    &children[i], points, next_params);
            }
        }
    }
}
```

### 点分类（统计阶段）

```cuda
__device__ void count_points_in_children(
    const Points &in_points, int *smem, 
    int range_begin, int range_end, float2 center) {
    
    // 初始化计数
    if (threadIdx.x < 4) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // 每个线程处理多个点
    for (int iter = range_begin + threadIdx.x; 
         iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);
        
        if (p.x < center.x && p.y >= center.y) {
            atomicAdd(&smem[0], 1);  // 左上
        } else if (p.x >= center.x && p.y >= center.y) {
            atomicAdd(&smem[1], 1);  // 右上
        } else if (p.x < center.x && p.y < center.y) {
            atomicAdd(&smem[2], 1);  // 左下
        } else {
            atomicAdd(&smem[3], 1);  // 右下
        }
    }
    __syncthreads();
}
```

### 深度分析

假设初始有 64 个均匀分布的点：

| 深度 | 节点数 | 每节点点数 |
| ---- | ------ | ---------- |
| 0    | 1      | 64         |
| 1    | 4      | 16         |
| 2    | 16     | 4          |
| 3    | 64     | 1          |

总共启动 1 + 4 + 16 = **21** 个子 kernel。

## 性能考虑

### 启动开销

子 kernel 启动有开销：

- 资源分配
- 参数传递
- 调度延迟

**建议**：只在工作量足够大时使用动态并行。

### 嵌套深度限制

CUDA 限制嵌套深度（通常 24 层）。

```cuda
// 检查支持的嵌套深度
int max_depth;
cudaDeviceGetAttribute(&max_depth, 
    cudaDevAttrMaxDeviceRuntimeSynchronizationDepth, 0);
```

### 内存消耗

每层递归消耗栈空间。深度递归可能导致栈溢出。

```cuda
// 增大栈大小
cudaDeviceSetLimit(cudaLimitStackSize, 8192);
```

### 何时使用动态并行

**适合使用**：

- 递归算法（树遍历、分治）
- 自适应细分（网格细化、LOD）
- 不规则数据（稀疏图、不平衡树）

**不适合使用**：

- 规则的数据并行（矩阵运算）
- 子 kernel 工作量很小
- 需要极致性能的场景

## 调试与最佳实践

### 错误处理

```cuda
__global__ void parent_kernel() {
    cudaError_t err;
    
    child_kernel<<<1, 32>>>();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Child kernel launch failed: %s\n", 
               cudaGetErrorString(err));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Child kernel execution failed: %s\n", 
               cudaGetErrorString(err));
    }
}
```

### 避免过度嵌套

```cuda
__global__ void recursive_kernel(int depth, int max_depth) {
    if (depth >= max_depth) {
        return;  // 停止递归
    }
    
    recursive_kernel<<<1, 32>>>(depth + 1, max_depth);
}
```

### 资源管理

```cuda
__global__ void parent_kernel() {
    float *temp;
    cudaMalloc(&temp, size);
    
    // 确保子 kernel 完成后再释放
    child_kernel<<<1, 32>>>(temp);
    cudaDeviceSynchronize();
    
    cudaFree(temp);
}
```

## 小结

第二十一章介绍了 CUDA 动态并行性这一高级特性：

**核心概念**：GPU kernel 可以直接启动子 kernel，无需返回 CPU。这使得递归算法和自适应计算可以完全在 GPU 上执行。

**语法要点**：

- 在 `__global__` 函数中使用 `<<<...>>>` 启动子 kernel
- `cudaDeviceSynchronize()` 等待子 kernel 完成
- 可以在设备端使用 `cudaMalloc`/`cudaFree`

**内存可见性**：

- 全局、常量、纹理内存父子共享
- 共享内存和局部内存不能传递给子 kernel

**流与并发**：

- 同一 Block 的默认流共享，子 kernel 串行执行
- 使用非阻塞流实现子 kernel 并发

**应用场景**：

- Bezier 曲线自适应细分：根据曲率动态决定采样密度
- 四叉树构建：递归划分空间

**性能考虑**：

- 启动开销不可忽视
- 深度递归消耗栈空间
- 配置启动池大小

动态并行性让 GPU 编程更加灵活，但要权衡使用——对于规则的数据并行任务，传统方式可能更高效。对于天然递归或自适应的问题，动态并行性是强大的工具。

## 🚀 下一步

- 实现 Bezier 曲线的自适应细分，体验动态并行性的优势
- 构建一个四叉树或八叉树，学习递归算法的 GPU 实现
- 探索其他递归算法：快速排序、归并排序、分治算法
- 学习动态并行性的性能调优：启动池配置、嵌套深度控制
- 对比动态并行性与传统方法的性能差异，理解适用场景
- 研究自适应网格细化（AMR）等高级应用

---

## 📚 参考资料

- PMPP 第四版 Chapter 21
- [第二十一章：CUDA动态并行性](https://smarter.xin/posts/pmmpp-chapter21-dynamic-parallelism/)
- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- NVIDIA. *CUDA C++ Programming Guide - Dynamic Parallelism*. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA. *CUDA Dynamic Parallelism*. <https://developer.nvidia.com/blog/cuda-dynamic-parallelism-api-principles/>

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
