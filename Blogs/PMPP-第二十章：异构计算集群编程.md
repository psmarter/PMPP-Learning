---
title: PMPP-第二十章：异构计算集群编程
date: 2026-01-24 15:26:44
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - MPI
  - 集群计算
  - CUDA流
  - 分布式计算
categories: 知识分享
cover: /img/PMPP.jpg
---

## 前言

第十九章总结了并行编程的思维方法。第二十章将视野扩展到**计算集群（Computing Cluster）**——多台计算机通过高速网络连接，每台计算机可能配备多个 GPU。这是当今超级计算机和数据中心的典型架构。本章讨论如何使用 **MPI（Message Passing Interface，消息传递接口）**与 CUDA 结合，实现跨节点的异构并行计算。掌握这些技术，就能编写可扩展到数千个 GPU 的大规模并行程序。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 异构计算集群架构

### 什么是异构集群

**异构集群**：由多个计算节点组成，每个节点包含：

- CPU（主机）
- 一个或多个 GPU（加速器）
- 本地内存
- 网络接口

节点之间通过高速网络（如 InfiniBand、NVLink）连接。

### 典型架构

```
┌─────────────────────────────────────────────────────────┐
│                    计算集群                              │
├─────────────────┬─────────────────┬─────────────────────┤
│    节点 0       │    节点 1       │    节点 N-1         │
│  ┌─────────┐   │  ┌─────────┐   │  ┌─────────┐        │
│  │  CPU    │   │  │  CPU    │   │  │  CPU    │        │
│  └────┬────┘   │  └────┬────┘   │  └────┬────┘        │
│       │        │       │        │       │             │
│  ┌────┴────┐   │  ┌────┴────┐   │  ┌────┴────┐        │
│  │GPU0│GPU1│   │  │GPU0│GPU1│   │  │GPU0│GPU1│        │
│  └─────────┘   │  └─────────┘   │  └─────────┘        │
└────────┬───────┴────────┬───────┴────────┬────────────┘
         │                │                │
         └────────────────┴────────────────┘
                    高速网络
```

### 编程挑战

1. **分布式内存**：每个节点有独立的内存空间，不能直接访问
2. **数据通信**：需要显式地在节点间传递数据
3. **同步协调**：多个进程需要协调工作
4. **故障容错**：单个节点故障不应导致整个计算崩溃

## MPI 基础

### 什么是 MPI

**MPI（Message Passing Interface）**：一种标准化的消息传递编程模型。

- 定义了进程间通信的 API
- 支持点对点通信和集合通信
- 与硬件无关，可移植性好

常见实现：OpenMPI、MPICH、Intel MPI。

### 基本概念

**进程（Process）**：MPI 程序的基本执行单元。每个进程有：

- 唯一的**秩（Rank）**：0 到 N-1
- 独立的地址空间
- 可以运行在不同的物理节点上

**通信子（Communicator）**：定义参与通信的进程组。

- `MPI_COMM_WORLD`：包含所有进程的默认通信子

### MPI 程序骨架

```c
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    
    // 初始化 MPI
    MPI_Init(&argc, &argv);
    
    // 获取当前进程的秩
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 获取进程总数
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    printf("进程 %d / %d\n", rank, size);
    
    // 计算和通信...
    
    // 结束 MPI
    MPI_Finalize();
    return 0;
}
```

**运行方式**：

```bash
mpirun -np 4 ./my_program   # 启动 4 个进程
```

## 点对点通信

### 基本发送和接收

```c
// 发送
MPI_Send(
    void *buf,          // 发送缓冲区
    int count,          // 元素个数
    MPI_Datatype dtype, // 数据类型（MPI_FLOAT, MPI_INT 等）
    int dest,           // 目标进程秩
    int tag,            // 消息标签
    MPI_Comm comm       // 通信子
);

// 接收
MPI_Recv(
    void *buf,          // 接收缓冲区
    int count,          // 最大元素个数
    MPI_Datatype dtype, // 数据类型
    int source,         // 源进程秩
    int tag,            // 消息标签
    MPI_Comm comm,      // 通信子
    MPI_Status *status  // 状态信息
);
```

### 阻塞与非阻塞

**阻塞通信**：函数返回时，操作已完成或缓冲区可安全复用。

- `MPI_Send`：可能阻塞直到接收方准备好（取决于实现）
- `MPI_Recv`：阻塞直到消息到达

**非阻塞通信**：函数立即返回，后续检查完成状态。

```c
MPI_Request request;

// 非阻塞发送
MPI_Isend(buf, count, dtype, dest, tag, comm, &request);

// 做其他事情...

// 等待完成
MPI_Wait(&request, &status);
```

### Send-Receive 组合

避免死锁的常用模式：

```c
MPI_Sendrecv(
    send_buf, send_count, send_type, dest, send_tag,
    recv_buf, recv_count, recv_type, source, recv_tag,
    comm, &status
);
```

同时发送和接收，系统自动处理顺序。

## MPI + CUDA 编程

### 基本策略

每个 MPI 进程管理一个或多个 GPU：

```c
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

// 每个进程选择不同的 GPU
int num_devices;
cudaGetDeviceCount(&num_devices);
cudaSetDevice(rank % num_devices);
```

### 数据流模式

典型的 MPI + CUDA 计算流程：

```
1. MPI 进程接收输入数据（主机内存）
2. 复制数据到 GPU（cudaMemcpy H2D）
3. GPU 计算（kernel）
4. 复制结果到主机（cudaMemcpy D2H）
5. MPI 进程发送结果
```

### 示例：分布式向量加法

```c
void distributed_vector_add(float *a, float *b, float *c, int n, int rank, int size) {
    int local_n = n / size;
    int start = rank * local_n;
    
    // 分配 GPU 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, local_n * sizeof(float));
    cudaMalloc(&d_b, local_n * sizeof(float));
    cudaMalloc(&d_c, local_n * sizeof(float));
    
    // 复制本地数据到 GPU
    cudaMemcpy(d_a, a + start, local_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b + start, local_n * sizeof(float), cudaMemcpyHostToDevice);
    
    // GPU 计算
    int block_size = 256;
    int grid_size = (local_n + block_size - 1) / block_size;
    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, local_n);
    
    // 复制结果回主机
    cudaMemcpy(c + start, d_c, local_n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // 收集所有结果到进程 0
    MPI_Gather(c + start, local_n, MPI_FLOAT,
               c, local_n, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
```

## Halo 交换与边界通信

### 什么是 Halo

在模板计算（stencil）、有限差分等应用中，每个点的计算依赖于邻近点。

当数据分布在多个进程时，边界点的计算需要**相邻进程的数据**。

**Halo（光晕/幽灵区域）**：存储来自邻居进程的边界数据。

```
进程 0 的数据        进程 1 的数据
┌──────────────┐    ┌──────────────┐
│              │    │              │
│   内部区域   │    │   内部区域   │
│              │    │              │
├──────────────┤    ├──────────────┤
│  右边界 →→→  │ ↔  │ ←←← 左 Halo │
│ (发送给 P1)  │    │ (来自 P0)   │
└──────────────┘    └──────────────┘
```

### 3D 模板计算示例

以 25 点模板为例（每个方向延伸 4 个点）：

```c
// 计算每个进程需要多少 Halo 点
int halo_size = 4;  // 每侧 4 层
int num_halo_points = dimx * dimy * halo_size;

// 分配包含 Halo 的数据
int total_z = local_dimz + 2 * halo_size;
float *data = malloc(dimx * dimy * total_z * sizeof(float));
```

### Halo 交换实现

```c
void exchange_halos(float *data, int dimx, int dimy, int dimz,
                     int left_neighbor, int right_neighbor) {
    int halo_size = 4;
    int num_halo_points = dimx * dimy * halo_size;
    
    float *left_send = data + num_halo_points;  // 左边界数据
    float *right_send = data + dimx * dimy * (dimz - halo_size);  // 右边界数据
    float *left_recv = data;  // 左 Halo 接收区
    float *right_recv = data + dimx * dimy * (dimz + halo_size);  // 右 Halo 接收区
    
    MPI_Status status;
    
    // 发送到左邻居，从右邻居接收
    MPI_Sendrecv(left_send, num_halo_points, MPI_FLOAT, left_neighbor, 0,
                 right_recv, num_halo_points, MPI_FLOAT, right_neighbor, 0,
                 MPI_COMM_WORLD, &status);
    
    // 发送到右邻居，从左邻居接收
    MPI_Sendrecv(right_send, num_halo_points, MPI_FLOAT, right_neighbor, 1,
                 left_recv, num_halo_points, MPI_FLOAT, left_neighbor, 1,
                 MPI_COMM_WORLD, &status);
}
```

## 计算与通信重叠

### 问题

通信需要时间，如果先计算完再通信，GPU 会空闲等待。

### 解决方案

**思路**：重叠计算与通信。

1. 先计算边界区域（通信需要的数据）
2. 边界计算完成后，开始通信
3. 同时计算内部区域
4. 通信完成后，所有计算都完成了

### CUDA 流实现

```cuda
cudaStream_t stream_boundary, stream_internal;
cudaStreamCreate(&stream_boundary);
cudaStreamCreate(&stream_internal);

// 阶段 1：计算边界（在 stream_boundary 中）
stencil_kernel<<<grid_boundary, block, 0, stream_boundary>>>(
    d_output + left_offset, d_input + left_offset, dimx, dimy, 12);
stencil_kernel<<<grid_boundary, block, 0, stream_boundary>>>(
    d_output + right_offset, d_input + right_offset, dimx, dimy, 12);

// 阶段 2：同时进行——
// 2a: 计算内部区域（在 stream_internal 中）
stencil_kernel<<<grid_internal, block, 0, stream_internal>>>(
    d_output + internal_offset, d_input + internal_offset, dimx, dimy, dimz - 8);

// 2b: 复制边界到主机，准备发送
cudaMemcpyAsync(h_left_boundary, d_output + boundary_left,
                num_halo_bytes, cudaMemcpyDeviceToHost, stream_boundary);
cudaMemcpyAsync(h_right_boundary, d_output + boundary_right,
                num_halo_bytes, cudaMemcpyDeviceToHost, stream_boundary);

// 等待边界复制完成
cudaStreamSynchronize(stream_boundary);

// 阶段 3：MPI 通信
MPI_Sendrecv(h_left_boundary, num_halo_points, MPI_FLOAT, left_neighbor, 0,
             h_right_halo, num_halo_points, MPI_FLOAT, right_neighbor, 0,
             MPI_COMM_WORLD, &status);
MPI_Sendrecv(h_right_boundary, num_halo_points, MPI_FLOAT, right_neighbor, 1,
             h_left_halo, num_halo_points, MPI_FLOAT, left_neighbor, 1,
             MPI_COMM_WORLD, &status);

// 复制 Halo 回 GPU
cudaMemcpyAsync(d_output + left_halo_offset, h_left_halo,
                num_halo_bytes, cudaMemcpyHostToDevice, stream_boundary);
cudaMemcpyAsync(d_output + right_halo_offset, h_right_halo,
                num_halo_bytes, cudaMemcpyHostToDevice, stream_boundary);

// 等待所有操作完成
cudaDeviceSynchronize();
```

### 时间线分析

```
                    时间 →
stream_boundary:  [边界计算][D2H][       ][H2D]
stream_internal:  [           内部计算          ]
MPI 通信:         [      ][Sendrecv][      ]
                           ↑
                     重叠区域：内部计算
                     与 MPI 通信同时进行
```

## CUDA-Aware MPI

### 传统方式的问题

```c
// 传统方式：必须经过主机内存
cudaMemcpy(h_buf, d_buf, size, cudaMemcpyDeviceToHost);  // GPU → CPU
MPI_Send(h_buf, count, MPI_FLOAT, dest, tag, comm);       // CPU → 网络
// 接收端
MPI_Recv(h_buf, count, MPI_FLOAT, src, tag, comm, &status);  // 网络 → CPU  
cudaMemcpy(d_buf, h_buf, size, cudaMemcpyHostToDevice);      // CPU → GPU
```

**问题**：额外的内存拷贝开销。

### CUDA-Aware MPI

**CUDA-Aware MPI**：MPI 实现能直接识别 GPU 指针。

```c
// 直接传递 GPU 指针——不需要手动拷贝
MPI_Send(d_buf, count, MPI_FLOAT, dest, tag, comm);
MPI_Recv(d_buf, count, MPI_FLOAT, src, tag, comm, &status);
```

MPI 库自动处理：

- 通过 GPUDirect RDMA 直接 GPU 到 GPU 传输
- 如果不支持，自动回退到经过主机的方式

### 环境配置

```bash
# 编译时链接 CUDA-Aware MPI
mpicc -o my_prog my_prog.c -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcudart

# 运行前设置
export UCX_RNDV_SCHEME=cuda
export UCX_TLS=rc,cuda_copy,cuda_ipc
```

### 使用 CUDA-Aware MPI 改写

```c
// 无需 host buffer，直接使用 device buffer
MPI_Sendrecv(d_output + boundary_left, num_halo_points, MPI_FLOAT, left_neighbor, 0,
             d_output + right_halo_offset, num_halo_points, MPI_FLOAT, right_neighbor, 0,
             MPI_COMM_WORLD, &status);
```

**优势**：

- 减少内存拷贝
- 更低延迟
- 代码更简洁

## 数据服务器模式

### 问题

大规模集群中，I/O 可能成为瓶颈。每个计算节点都从存储读数据会导致争用。

### 解决方案

**数据服务器模式**：一个进程专门负责 I/O，其他进程专门计算。

```
┌──────────────────────────────────────────────────────┐
│  数据服务器 (Rank N-1)                               │
│  - 读取输入数据                                      │
│  - 分发数据给计算节点                                │
│  - 收集计算结果                                      │
│  - 写入输出                                          │
└──────────────────────────────────────────────────────┘
         ↓ 分发           ↑ 收集
┌─────────┬─────────┬─────────┬─────────┐
│ Rank 0  │ Rank 1  │ Rank 2  │   ...   │
│ 计算    │ 计算    │ 计算    │         │
└─────────┴─────────┴─────────┴─────────┘
```

### 实现

```c
void data_server(int dimx, int dimy, int dimz, int nreps) {
    int np, num_comp_nodes;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    num_comp_nodes = np - 1;
    
    // 分配并初始化数据
    float *input = malloc(dimx * dimy * dimz * sizeof(float));
    float *output = malloc(dimx * dimy * dimz * sizeof(float));
    initialize_data(input, dimx, dimy, dimz);
    
    // 计算每个节点的数据量
    int slice_per_node = dimz / num_comp_nodes;
    int halo_size = 4;
    
    // 分发数据给计算节点
    for (int p = 0; p < num_comp_nodes; p++) {
        int start_z = p * slice_per_node - (p > 0 ? halo_size : 0);
        int num_slices = slice_per_node + (p > 0 ? halo_size : 0) 
                                        + (p < num_comp_nodes - 1 ? halo_size : 0);
        int num_points = dimx * dimy * num_slices;
        
        MPI_Send(input + start_z * dimx * dimy, num_points, MPI_FLOAT,
                 p, 0, MPI_COMM_WORLD);
    }
    
    // 等待计算完成
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 收集结果
    for (int p = 0; p < num_comp_nodes; p++) {
        int offset = p * slice_per_node * dimx * dimy;
        int num_points = slice_per_node * dimx * dimy;
        
        MPI_Recv(output + offset, num_points, MPI_FLOAT,
                 p, DATA_COLLECT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // 保存结果
    save_output(output, dimx, dimy, dimz);
    
    free(input);
    free(output);
}
```

## 完整示例：MPI + CUDA 模板计算

### 程序结构

```c
int main(int argc, char *argv[]) {
    int pid, np;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    if (pid < np - 1) {
        // 计算节点：设置 GPU 并计算
        int device = pid % num_devices;
        cudaSetDevice(device);
        compute_node_stencil(dimx, dimy, dimz / (np - 1), nreps);
    } else {
        // 数据服务器：I/O 和数据分发
        data_server(dimx, dimy, dimz, nreps);
    }
    
    MPI_Finalize();
    return 0;
}
```

### 计算节点实现

```c
void compute_node_stencil(int dimx, int dimy, int dimz, int nreps) {
    int pid, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    int server = np - 1;
    int left_neighbor = (pid > 0) ? (pid - 1) : MPI_PROC_NULL;
    int right_neighbor = (pid < np - 2) ? (pid + 1) : MPI_PROC_NULL;
    
    // 分配内存
    int halo = 4;
    int total_z = dimz + 2 * halo;
    size_t num_bytes = dimx * dimy * total_z * sizeof(float);
    
    float *h_input = malloc(num_bytes);
    float *h_output = malloc(num_bytes);
    float *d_input, *d_output;
    cudaMalloc(&d_input, num_bytes);
    cudaMalloc(&d_output, num_bytes);
    
    // Halo 缓冲区（固定内存，加速传输）
    float *h_left_boundary, *h_right_boundary;
    float *h_left_halo, *h_right_halo;
    size_t halo_bytes = dimx * dimy * halo * sizeof(float);
    cudaHostAlloc(&h_left_boundary, halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_boundary, halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_left_halo, halo_bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_right_halo, halo_bytes, cudaHostAllocDefault);
    
    // 创建 CUDA 流
    cudaStream_t stream_boundary, stream_internal;
    cudaStreamCreate(&stream_boundary);
    cudaStreamCreate(&stream_internal);
    
    // 从数据服务器接收初始数据
    MPI_Status status;
    MPI_Recv(h_input, dimx * dimy * total_z, MPI_FLOAT,
             server, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    cudaMemcpy(d_input, h_input, num_bytes, cudaMemcpyHostToDevice);
    
    // 迭代计算
    for (int iter = 0; iter < nreps; iter++) {
        // 阶段 1：边界计算
        launch_boundary_kernel(d_output, d_input, dimx, dimy, stream_boundary);
        
        // 阶段 2：内部计算（与通信重叠）
        launch_internal_kernel(d_output, d_input, dimx, dimy, dimz, stream_internal);
        
        // 复制边界到主机
        copy_boundary_to_host(d_output, h_left_boundary, h_right_boundary,
                               dimx, dimy, halo, stream_boundary);
        cudaStreamSynchronize(stream_boundary);
        
        // Halo 交换
        MPI_Sendrecv(h_left_boundary, dimx * dimy * halo, MPI_FLOAT, left_neighbor, iter,
                     h_right_halo, dimx * dimy * halo, MPI_FLOAT, right_neighbor, iter,
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(h_right_boundary, dimx * dimy * halo, MPI_FLOAT, right_neighbor, iter,
                     h_left_halo, dimx * dimy * halo, MPI_FLOAT, left_neighbor, iter,
                     MPI_COMM_WORLD, &status);
        
        // 复制 Halo 回 GPU
        copy_halo_to_device(d_output, h_left_halo, h_right_halo,
                             dimx, dimy, dimz, halo, stream_boundary);
        
        cudaDeviceSynchronize();
        
        // 交换输入输出指针
        float *temp = d_output;
        d_output = d_input;
        d_input = temp;
    }
    
    // 发送结果给数据服务器
    cudaMemcpy(h_output, d_input, num_bytes, cudaMemcpyDeviceToHost);
    MPI_Send(h_output + dimx * dimy * halo, dimx * dimy * dimz, MPI_FLOAT,
             server, DATA_COLLECT, MPI_COMM_WORLD);
    
    // 清理
    free(h_input);
    free(h_output);
    cudaFreeHost(h_left_boundary);
    cudaFreeHost(h_right_boundary);
    cudaFreeHost(h_left_halo);
    cudaFreeHost(h_right_halo);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream_boundary);
    cudaStreamDestroy(stream_internal);
}
```

## 性能优化

### 通信优化

| 技术           | 描述                              | 效果              |
| -------------- | --------------------------------- | ----------------- |
| 非阻塞通信     | 使用 `MPI_Isend`/`MPI_Irecv`      | 重叠通信与计算    |
| 集合通信       | 使用 `MPI_Allreduce` 而非循环 P2P | 利用优化的算法    |
| CUDA-Aware MPI | 直接传递 GPU 指针                 | 减少内存拷贝      |
| 固定内存       | `cudaHostAlloc`                   | 加速 H2D/D2H 传输 |

### 负载均衡

确保每个节点的工作量大致相等：

```c
// 处理不能整除的情况
int base_slices = dimz / num_nodes;
int remainder = dimz % num_nodes;

for (int p = 0; p < num_nodes; p++) {
    int slices = base_slices + (p < remainder ? 1 : 0);
    // 分配 slices 给节点 p
}
```

### 可扩展性分析

对于 25 点模板计算：

| 节点数 | 通信量（每节点） | 计算量（每节点） | 计算/通信比 |
| ------ | ---------------- | ---------------- | ----------- |
| 16     | 2×64×64×4 = 32K  | 64×64×128 = 512K | 16:1        |
| 64     | 2×64×64×4 = 32K  | 64×64×32 = 128K  | 4:1         |
| 256    | 2×64×64×4 = 32K  | 64×64×8 = 32K    | 1:1         |

**观察**：节点越多，通信开销占比越高。这是**强扩展**的典型特征。

## 常见问题与解决

### 死锁

**原因**：所有进程都在等待接收，没有进程发送。

```c
// 错误示例——会死锁！
if (rank == 0) {
    MPI_Recv(..., 1, ...);  // 等待 rank 1
    MPI_Send(..., 1, ...);
} else {
    MPI_Recv(..., 0, ...);  // 等待 rank 0
    MPI_Send(..., 0, ...);
}
```

**解决**：使用 `MPI_Sendrecv` 或非阻塞通信。

### GPU 内存不足

**原因**：每个节点分配的数据太多。

**解决**：

- 增加节点数
- 使用统一内存自动管理
- 分批处理

### 性能不佳

**诊断**：使用 Nsight Systems 分析 MPI + CUDA 程序。

```bash
nsys profile --trace=cuda,mpi mpirun -np 4 ./my_program
```

查看是否有：

- 过长的 MPI 等待时间
- 未重叠的计算和通信
- GPU 空闲时间

## 小结

第二十章扩展了并行编程的视野，从单 GPU 扩展到多节点集群：

**异构集群架构**：每个节点包含 CPU 和 GPU，节点间通过网络连接。分布式内存模型要求显式通信。

**MPI 基础**：消息传递编程模型。进程通过发送/接收消息通信。点对点通信（Send/Recv）和集合通信（Broadcast/Reduce）。

**MPI + CUDA**：每个 MPI 进程管理一个或多个 GPU。数据在主机内存和 GPU 内存之间传输，在进程间通过 MPI 传输。

**Halo 交换**：模板计算中，边界数据需要与邻居进程交换。使用 Sendrecv 避免死锁。

**计算与通信重叠**：利用 CUDA 流，边界计算完成后立即开始通信，同时进行内部计算。显著减少总执行时间。

**CUDA-Aware MPI**：MPI 库直接接受 GPU 指针，利用 GPUDirect 技术减少内存拷贝。

**数据服务器模式**：一个进程专门负责 I/O，减少存储争用。

掌握 MPI + CUDA 编程，你就能编写可扩展到数千 GPU 的应用程序——这是当今 AI 训练、科学计算、天气预报等领域的核心技术。

## 🚀 下一步

- 搭建一个简单的多节点 GPU 集群环境，配置 MPI 和 CUDA-Aware MPI
- 实现一个分布式矩阵乘法，学习数据分割和结果收集
- 掌握 Halo 交换模式，实现分布式模板计算（如热传导方程）
- 学习集合通信操作：Allreduce、Allgather、Alltoall
- 探索性能分析工具：Nsight Systems 分析 MPI + CUDA 程序的性能瓶颈
- 了解现代 HPC 框架：NCCL（多 GPU 通信）、Horovod（分布式深度学习）

---

## 📚 参考资料

- PMPP 第四版 Chapter 20
- [第二十章：异构计算集群编程](https://smarter.xin/posts/pmmpp-chapter20-heterogeneous-clusters/)
- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- MPI Forum. *MPI: A Message-Passing Interface Standard*. <https://www.mpi-forum.org/>
- NVIDIA. *CUDA-Aware MPI*. <https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/>

**学习愉快！** 🎓

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
