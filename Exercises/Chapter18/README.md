# 第十八章：静电势能图

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章介绍分子动力学中静电势能计算的 GPU 加速技术：

- 静电势能基础与库仑定律
- Scatter vs Gather 并行化策略
- 常量内存优化（原子数据存储）
- Thread Coarsening（每线程处理多点）
- Memory Coalescing（优化内存访问模式）
- 截断方法与空间分区（Cutoff / Cell List）

**相关博客笔记**：[第十八章：静电势能图](https://smarter.xin/posts/d7c6e6a8/)

---

## 💻 代码实现

### Exercise01 - 静电势能计算

实现书中所有静电势能计算 kernel，对应 Fig. 18.5、18.6、18.8、18.10。

**代码位置**：`Exercise01/`

**实现列表**：

| 实现 | 书中对应 | 特点 |
| ---- | -------- | ---- |
| `cenergySequential` | CPU 参考 | 串行遍历网格点和原子 |
| `cenergySequentialOptimized` | CPU 优化 | 先遍历原子（更好缓存利用） |
| `cenergyParallelScatter` | Fig. 18.5 | GPU Scatter：每线程一个原子 |
| `cenergyParallelGather` | Fig. 18.6 | GPU Gather：每线程一个网格点 |
| `cenergyParallelCoarsen` | Fig. 18.8 | Thread Coarsening |
| `cenergyParallelCoalescing` | Fig. 18.10 | Memory Coalescing 优化 |

**核心代码**：

```cuda
// Gather Kernel (Fig. 18.6) - 每个线程处理一个网格点
__global__ void cenergyGatherKernel(float* energygrid, dim3 grid_dim, 
                                    float gridspacing, float z, 
                                    int atoms_in_chunk, int chunk_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < grid_dim.x && j < grid_dim.y) {
        float x = gridspacing * (float)i;
        float y = gridspacing * (float)j;
        int k = (int)(z / gridspacing);
        
        float energy = 0.0f;
        for (int n = 0; n < atoms_in_chunk; n++) {
            float dx = x - atoms[n*4];
            float dy = y - atoms[n*4 + 1];
            float dz = z - atoms[n*4 + 2];
            float charge = atoms[n*4 + 3];
            energy += charge / sqrtf(dx*dx + dy*dy + dz*dz);
        }
        
        energygrid[grid_dim.x*grid_dim.y*k + grid_dim.x*j + i] += energy;
    }
}
```

#### 运行 Exercise01

```bash
cd Exercise01
make
make run
```

#### 预期输出

```text
================================================================
  第十八章：静电势能图
  Electrostatic Potential Map - Multiple Implementations
================================================================

正确性验证
================================================================

网格尺寸: 64 x 64 x 32
原子数量: 1000

1. 计算 CPU 参考结果...
2. 测试 GPU Scatter (Fig. 18.5)...
   ✅ 正确！
3. 测试 GPU Gather (Fig. 18.6)...
   ✅ 正确！
4. 测试 GPU Thread Coarsening (Fig. 18.8)...
   ✅ 正确！
5. 测试 GPU Memory Coalescing (Fig. 18.10)...
   ✅ 正确！

性能基准测试
================================================================

| 实现 | 时间 (ms) | 相对 CPU 加速 |
|------|-----------|---------------|
| CPU Sequential | 850.000 | 1.00x |
| GPU Scatter    | 45.000  | 18.89x |
| GPU Gather     | 12.000  | 70.83x |
| GPU Coarsen    | 8.500   | 100.00x |
| GPU Coalescing | 6.200   | 137.10x |
```

---

## 📖 练习题解答

### 练习 1: Fig. 18.6 主机代码

**题目**：完成 Fig. 18.6 中 Gather kernel 的主机代码，包括网格配置和 kernel 调用。

**解答**：

完整实现见 `Exercise01/solution.cu` 中的 `cenergyParallelGather()` 函数。

```cpp
void cenergyParallelGather(float* host_energygrid, dim3 grid_dim, float gridspacing,
                           float z, const float* host_atoms, int numatoms) {
    // 分配设备内存
    float* d_energygrid = NULL;
    size_t grid_size = grid_dim.x * grid_dim.y * grid_dim.z * sizeof(float);
    cudaMalloc((void**)&d_energygrid, grid_size);
    cudaMemset(d_energygrid, 0, grid_size);

    // 分块处理原子
    int num_chunks = (numatoms + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // 2D 线程块和网格配置
    dim3 threadsPerBlock(16, 16);  // 16×16 = 256 threads
    dim3 blocksPerGrid((grid_dim.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (grid_dim.y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int start_atom = chunk * CHUNK_SIZE;
        int atoms_in_chunk = min(CHUNK_SIZE, numatoms - start_atom);

        // 复制原子数据到常量内存
        size_t chunk_bytes = atoms_in_chunk * 4 * sizeof(float);
        cudaMemcpyToSymbol(atoms, &host_atoms[start_atom * 4], chunk_bytes);

        // 启动 kernel
        cenergyGatherKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_energygrid, grid_dim, gridspacing, z, atoms_in_chunk, start_atom);

        cudaDeviceSynchronize();
    }

    // 复制结果回主机
    cudaMemcpy(host_energygrid, d_energygrid, grid_size, cudaMemcpyDeviceToHost);
    cudaFree(d_energygrid);
}
```

**关键点**：

1. **2D 线程块**：使用 16×16 的线程块处理 2D 网格切片
2. **分块处理**：原子数据按 `CHUNK_SIZE` 分块，每块复制到常量内存
3. **常量内存**：利用常量缓存，所有线程访问相同原子数据时只需一次读取

---

### 练习 2: 操作数对比分析

**题目**：比较 Fig. 18.6（原始）和 Fig. 18.8（Thread Coarsening）的操作数，COARSEN_FACTOR = 8。

**Fig. 18.6 原始 Kernel**（每迭代处理 1 个网格点，需 8 次迭代等效）：

| 操作类型 | 每次迭代 | 8 次迭代总计 |
|----------|----------|--------------|
| 内存读取 | 4（atoms[n], atoms[n+1], atoms[n+2], atoms[n+3]） | 32 |
| 算术操作 | 11（3 减法 + 3 乘法 + 3 加法 + 1 sqrt + 1 除法） | 88 |
| 分支 | 1（循环条件） | 8 |

**Fig. 18.8 Thread Coarsening Kernel**（每迭代处理 8 个网格点）：

| 操作类型 | 外循环 | 内循环 (×8) | 总计 |
|----------|--------|-------------|------|
| 内存读取 | 3 (dy, dz, charge) | 1 (atoms[n]) | 3 + 8 = **11** |
| 算术操作 | 5 (2 减法 + 2 乘法 + 1 加法) | 7 (1 乘 + 1 减 + 1 乘 + 1 加 + 1 sqrt + 1 除 + 1 加) | 5 + 56 = **61** |
| 分支 | 1 | 2 (循环 + 边界检查) | 1 + 16 = **17** |

**对比总结**：

| 操作类型 | Fig. 18.6 | Fig. 18.8 | 变化 |
|----------|-----------|-----------|------|
| 内存读取 | 32 | 11 | **-65.6%** |
| 算术操作 | 88 | 61 | **-30.7%** |
| 分支 | 8 | 17 | **+112.5%** |

**结论**：Thread Coarsening 显著减少了内存读取和算术操作，但增加了分支。整体上，由于内存读取是主要瓶颈，性能提升明显。

---

### 练习 3: Thread Coarsening 的缺点

**题目**：给出 Section 18.3 中增加每线程工作量的两个潜在缺点。

**解答**：

1. **寄存器压力增加**：
   - 每线程需要更多寄存器存储中间结果（如 `energies[COARSEN_FACTOR]` 数组）
   - 如果寄存器使用超过硬件限制，会降低 occupancy（SM 上可并发的线程数）
   - 可能导致寄存器溢出到本地内存，严重影响性能

2. **并行度降低风险**：
   - COARSEN_FACTOR 过大时，启动的线程数减少
   - 如果线程数不足以充分利用 GPU 的所有 SM，会导致资源闲置
   - 代码变得接近串行，无法充分发挥 GPU 并行能力

**最佳实践**：选择适当的 COARSEN_FACTOR（通常 4-16），平衡寄存器使用和并行度。

---

### 练习 4: Bin 邻域列表的控制分歧

**题目**：使用 Fig. 18.13 解释当线程处理邻域列表中的 bin 时如何产生控制分歧。

**解答**：

在使用截断方法（cutoff）和空间分区（Cell/Bin List）时：

```cpp
// 遍历邻近 27 个 bin
for (int dcx = -1; dcx <= 1; dcx++) {
    for (int dcy = -1; dcy <= 1; dcy++) {
        for (int dcz = -1; dcz <= 1; dcz++) {
            int bin_idx = ...;
            int num_atoms_in_bin = bin_count[bin_idx];  // 不同 bin 原子数不同！
            
            for (int i = 0; i < num_atoms_in_bin; i++) {  // 分歧来源
                // 处理原子
            }
        }
    }
}
```

**控制分歧产生原因**：

1. **bin 中原子数不均匀**：
   - 不同网格点的邻近 bin 可能包含不同数量的原子
   - 同一 warp 中的线程需要迭代不同次数

2. **分歧情况**：
   - 假设 warp 中 32 个线程处理 32 个不同网格点
   - 线程 0 的邻近 bin 有 50 个原子，线程 1 的邻近 bin 有 10 个原子
   - 当迭代到第 11-50 次时，只有部分线程活跃

**缓解策略**：

1. **填充虚拟原子**：用电荷为 0 的虚拟原子填充 bin，使所有 bin 原子数相同
   - 缺点：浪费内存和计算

2. **动态负载均衡**：使用 warp 级归约或 cooperative groups
3. **排序优化**：按邻域原子数对网格点排序，相似工作量的点分配到同一 warp

---

## 📁 项目结构

```text
Chapter18/
├── README.md           # 本文档
└── Exercise01/         # 静电势能计算
    ├── solution.h      # 头文件
    ├── solution.cu     # CUDA 实现
    ├── test.cpp        # 测试程序
    └── Makefile        # 编译配置
```

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0+
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: NVIDIA 显卡（计算能力 3.5+）

---

## 💡 学习建议

1. **理解 Scatter vs Gather**：
   - Scatter：原子中心视角，需要原子操作
   - Gather：网格点中心视角，无冲突但重复读原子

2. **常量内存的作用**：
   - 所有线程访问相同地址时，广播机制只需一次读取
   - 原子数据在所有线程间共享，非常适合常量内存

3. **Thread Coarsening 权衡**：
   - 减少冗余计算（dy、dz 只算一次）
   - 但增加寄存器使用，需要调优

4. **Memory Coalescing**：
   - 确保 warp 内线程访问连续内存
   - 对性能影响可达 10 倍以上

5. **实际应用优化**：对于大规模分子系统，考虑使用截断方法（cutoff）和空间分区（Cell List）来减少计算量，只计算距离小于截断半径的原子对；使用 `__restrict__` 关键字帮助编译器优化内存访问；在生产环境中，可以使用 NAMD、GROMACS 等成熟的分子动力学软件，它们已经实现了高度优化的 GPU 加速

---

## 📚 参考资料

- PMPP 第四版 Chapter 18
- [GitHub参考仓库](https://github.com/tugot17/pmpp/tree/main/chapter-18)
- [第十八章：静电势能图](https://smarter.xin/posts/d7c6e6a8/)

**学习愉快！** 🎓
