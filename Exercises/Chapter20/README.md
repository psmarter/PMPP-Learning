# 第二十章：异构计算集群编程

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章介绍 MPI + CUDA 编程，实现跨多节点的 GPU 集群计算：

- MPI 基础（进程、通信子、点对点通信）
- MPI + CUDA 编程模式
- Halo 交换与边界通信
- 计算与通信重叠（CUDA 流）
- CUDA-Aware MPI（GPU 直接通信）
- 数据服务器模式

**相关博客笔记**：[第二十章：异构计算集群编程](https://smarter.xin/posts/9506dbb9/)

**重要提示**：本章代码需要 **MPI 环境**才能编译运行。

安装方法：

- Ubuntu/Debian: `sudo apt install openmpi-bin libopenmpi-dev`
- CentOS/RHEL: `sudo yum install openmpi openmpi-devel`

---

## 💻 代码实现

### Exercise01 - MPI+CUDA 分布式模板计算

实现书中的分布式 3D 模板计算，包含数据服务器模式和 Halo 交换。

**代码位置**：`Exercise01/`

**实现组件**：

| 组件 | 说明 |
| ---- | ---- |
| `stencil_kernel` | 5 点模板 CUDA 核函数 |
| `data_server()` | 数据服务器进程（分发/收集） |
| `compute_node_stencil()` | 计算节点（Halo 交换 + 流重叠） |
| `call_stencil_kernel()` | 内核启动封装 |

**关键特性**：

- **数据服务器模式**：1 个进程专门负责 I/O
- **Halo 交换**：使用 `MPI_Sendrecv` 交换边界数据
- **计算-通信重叠**：stream0 处理边界，stream1 处理内部

```bash
cd Exercise01
make
mpirun -np 3 ./stencil_mpi
```

#### 预期输出

```text
================================================================
  第二十章：异构计算集群编程
  MPI + CUDA Distributed Stencil Computation
================================================================

配置信息:
  网格尺寸: 48 x 48 x 40
  迭代次数: 10
  MPI 进程数: 3 (2 计算节点 + 1 数据服务器)
  Halo 大小: 4

Output computed for grid 48 x 48 x 40
First few output values: 23.456 -14.789 40.123 33.456 71.789
```

---

## 📖 练习题解答

### 练习 1: 25 点模板计算分析

**题目**：假设 25 点模板计算应用于 64×64×2048 网格，使用 17 个 MPI 进程（16 计算 + 1 数据服务器），沿 z 轴划分。

#### (a) 每个计算进程输出多少网格点？

**分析**：

- 网格沿 z 轴划分：每进程 2048/16 = 128 层
- 25 点模板每方向延伸 4 点，损失 8 点（x、y 各损失 8）
- 内部进程：`(64-8) × (64-8) × 128 = 56 × 56 × 128 = 401,408` 点
- 边缘进程：`(64-8) × (64-8) × (128-4) = 56 × 56 × 124 = 388,864` 点

#### (b) 需要多少 Halo 点？

**i. 内部计算进程：**

- 左右各 4 层 Halo：`(64 × 64 × 4) × 2 = 16,384 × 2 = 32,768` Halo 点

**ii. 边缘计算进程：**

- 只有一侧 Halo：`64 × 64 × 4 = 16,384` Halo 点

#### (c) Fig. 20.12 阶段 1 计算多少边界点？

**i. 内部计算进程：**

- 左右边界各 4 层：`((64-8) × (64-8) × 4) × 2 = 12,544 × 2 = 25,088` 点

**ii. 边缘计算进程：**

- 只有一侧边界：`(64-8) × (64-8) × 4 = 12,544` 点

#### (d) Fig. 20.12 阶段 2 计算多少内部点？

**i. 内部计算进程：**

- 总输出 401,408 - 阶段 1 的 25,088 = **376,320** 点
- 验证：`56 × 56 × 120 = 376,320` ✓

**ii. 边缘计算进程：**

- 总输出 388,864 - 阶段 1 的 12,544 = **376,320** 点

#### (e) Fig. 20.12 阶段 2 发送多少字节？

**i. 内部计算进程：**

- 左右各发送 4 层 Halo：`(4 × 64 × 64 × 4 bytes) × 2 = 65,536 × 2 = 131,072` 字节

**ii. 边缘计算进程：**

- 只发送一侧：`65,536` 字节

---

### 练习 2: MPI_Send 数据元素大小

**题目**：如果 `MPI_Send(ptr_a, 1000, MPI_FLOAT, 2000, 4, MPI_COMM_WORLD)` 传输 4000 字节，每个数据元素多大？

**解答**：

- 传输 4000 字节，1000 个元素
- 每元素 = 4000 / 1000 = **4 字节**
- 答案：**C. 4 bytes**

---

### 练习 3: MPI 特性判断

**a. MPI_Send() 默认是阻塞的？**

- **部分正确**：对小消息可能缓冲后立即返回，大消息阻塞直到接收方准备好

**b. MPI_Recv() 默认是阻塞的？**

- **正确**：`MPI_Recv` 始终阻塞直到消息到达

**c. MPI 消息必须至少 128 字节？**

- **错误**：可以发送任意大小消息（包括 0 字节）

**d. MPI 进程可以通过共享内存访问同一变量？**

- **错误**：MPI 是分布式内存模型，每个进程有独立地址空间

---

### 练习 4: CUDA-Aware MPI 代码改写

**题目**：修改示例代码，移除 `cudaMemcpyAsync()`，使用 GPU 内存地址直接传递给 MPI。

**改写后的代码**：

```cpp
void compute_node_stencil(int dimx, int dimy, int dimz, int nreps) {
    // ... 初始化代码省略 ...
    
    for (int i = 0; i < nreps; i++) {
        // 阶段 1：计算边界
        call_stencil_kernel(d_output + left_stage1_offset,
                            d_input + left_stage1_offset, dimx, dimy, 12, stream0);
        call_stencil_kernel(d_output + right_stage1_offset,
                            d_input + right_stage1_offset, dimx, dimy, 12, stream0);
        
        // 阶段 2：计算内部
        call_stencil_kernel(d_output + stage2_offset,
                            d_input + stage2_offset, dimx, dimy, dimz, stream1);
        
        // 移除：cudaMemcpyAsync 到主机
        // 移除：cudaStreamSynchronize(stream0)
        
        // CUDA-Aware MPI：直接使用 GPU 指针
        MPI_Sendrecv(d_output + num_halo_points, num_halo_points, MPI_FLOAT,
                     left_neighbor, i,
                     d_output + right_halo_offset, num_halo_points, MPI_FLOAT,
                     right_neighbor, i, MPI_COMM_WORLD, &status);
        
        MPI_Sendrecv(d_output + right_stage1_offset + num_halo_points, num_halo_points, MPI_FLOAT,
                     right_neighbor, i,
                     d_output + left_halo_offset, num_halo_points, MPI_FLOAT,
                     left_neighbor, i, MPI_COMM_WORLD, &status);
        
        // 移除：cudaMemcpyAsync 回 GPU
        
        cudaDeviceSynchronize();
        
        float* temp = d_output;
        d_output = d_input;
        d_input = temp;
    }
    
    // ... 清理代码省略 ...
}
```

**优势**：

- 减少 CPU-GPU 内存拷贝
- 利用 GPUDirect RDMA 实现 GPU 直接通信
- 代码更简洁

---

## 📁 项目结构

```text
Chapter20/
├── README.md               # 本文档
└── Exercise01/             # MPI+CUDA 分布式模板计算
    ├── solution.h          # 头文件
    ├── solution.cu         # CUDA+MPI 实现
    ├── test.cpp            # 主程序
    └── Makefile            # MPI+CUDA 编译配置
```

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0+
- **MPI**: OpenMPI 或 MPICH
- **编译器**: GCC 7.5+ / NVCC

---

## 💡 学习建议

1. **理解 MPI 基本概念**：
   - 进程秩（Rank）、通信子（Communicator）
   - 阻塞 vs 非阻塞通信

2. **Halo 交换模式**：
   - 理解为什么需要 Halo
   - 使用 `MPI_Sendrecv` 避免死锁

3. **计算-通信重叠**：
   - CUDA 流并行计算和通信
   - 最大化 GPU 利用率

4. **CUDA-Aware MPI**：
   - 直接传递 GPU 指针
   - 减少内存拷贝开销

5. **调试与性能分析**：使用 `MPI_Barrier` 和 `cudaDeviceSynchronize` 进行同步调试，但要注意这些会降低性能；使用 `MPI_Allreduce` 进行全局归约操作（如计算全局误差）；性能分析时关注通信与计算的比例，理想情况下通信时间应远小于计算时间；对于大规模集群，考虑使用 NCCL（NVIDIA Collective Communications Library）进行高效的集合通信

---

## 📚 参考资料

- PMPP 第四版 Chapter 20
- [GitHub参考仓库](https://github.com/tugot17/pmpp/tree/main/chapter-20)
- [第二十章：异构计算集群编程](https://smarter.xin/posts/9506dbb9/)
- [NVIDIA CUDA-Aware MPI](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/)

**学习愉快！** 🎓
