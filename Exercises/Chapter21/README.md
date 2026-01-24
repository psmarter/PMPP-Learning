# 第二十一章：CUDA 动态并行性

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章介绍 CUDA 动态并行性，允许 GPU kernel 直接启动子 kernel：

- 动态并行性概念与语法
- 设备端同步与内存管理
- 内存可见性规则
- 流与并发控制
- 启动池配置
- 应用：Bezier 曲线自适应细分
- 应用：四叉树递归构建

**相关博客笔记**：[第二十一章：CUDA动态并行性](https://smarter.xin/posts/e519c4bc/)

**重要提示**：编译要求：动态并行需要计算能力 3.5+ 的 GPU

```bash
nvcc -arch=sm_35 -rdc=true my_program.cu -lcudadevrt
```

---

## 💻 代码实现

### Exercise01 - Bezier 曲线自适应细分

根据曲率动态决定采样密度，对比静态与动态并行版本。

**代码位置**：`Exercise01/`

| 实现 | 说明 |
| ---- | ---- |
| `computeBezierLines_static()` | 静态版本：循环处理 |
| `computeBezierLines_dynamic()` | 动态并行：父 kernel 启动子 kernel |
| `computeBezierLine_child()` | 子 kernel：计算细分点 |

```bash
cd Exercise01 && make && make run
```

![Bezier 曲线对比](bezier_comparison_static_vs_dynamic.png)

#### 预期输出示例

```text
================================================================
  第二十一章：CUDA 动态并行性
  Exercise 01: Bezier 曲线自适应细分
================================================================

GPU: NVIDIA GeForce RTX 3080 (计算能力 8.6)

测试参数:
  曲线数量: 1000
  随机种子: 42

运行静态版本...
  静态版本耗时: 2.456 ms
运行动态并行版本...
  动态并行版本耗时: 1.823 ms

验证结果...
  ✓ 结果匹配！

================================================================
性能对比:
  静态版本:     2.456 ms
  动态并行版本: 1.823 ms
  加速比:       1.35x
================================================================

样本结果（前 5 条曲线）:
  曲线 0: 12 顶点
  曲线 1: 8 顶点
  曲线 2: 16 顶点
  曲线 3: 10 顶点
  曲线 4: 14 顶点
```

**输出说明**：
- **GPU信息**：显示GPU型号和计算能力（需要3.5+支持动态并行）
- **性能对比**：动态并行版本通常比静态版本更快，因为可以根据曲率自适应调整采样密度
- **加速比**：动态并行版本的性能提升倍数
- **样本结果**：显示前5条曲线的顶点数，反映不同曲率下的自适应细分效果

---

### Exercise02 - 四叉树递归构建

使用动态并行递归划分 2D 空间，构建四叉树。

**代码位置**：`Exercise02/`

| 实现 | 说明 |
| ---- | ---- |
| `build_quadtree_kernel()` | 递归主 kernel |
| `count_points_in_children()` | 统计象限点数 |
| `reorder_points()` | 点重排 |
| `prepare_children()` | 准备子节点 |

```bash
cd Exercise02 && make && make run
```

![四叉树可视化](quadtree.png)

#### 预期输出示例

```text
================================================================
  第二十一章：CUDA 动态并行性
  Exercise 02: 四叉树递归构建
================================================================

GPU: NVIDIA GeForce RTX 3080 (计算能力 8.6)

测试参数:
  点数量: 1000
  最大深度: 5
  节点最小点数: 4
  随机种子: 42

构建四叉树...
  ✓ 四叉树构建完成
  耗时: 3.142 ms

================================================================
结果统计:
  输入点数: 1000
  输出点数: 1000
  最大深度: 5
  理论最大子 kernel 数: 341
================================================================

样本点（前 5 个）:
  点 0: (0.3745, 0.9507) -> (0.3745, 0.9507)
  点 1: (0.7319, 0.5987) -> (0.7319, 0.5987)
  点 2: (0.1560, 0.1560) -> (0.1560, 0.1560)
  点 3: (0.0581, 0.8662) -> (0.0581, 0.8662)
  点 4: (0.6011, 0.7081) -> (0.6011, 0.7081)
```

**输出说明**：
- **GPU信息**：显示GPU型号和计算能力（需要3.5+支持动态并行）
- **测试参数**：
  - `点数量`：输入的点总数
  - `最大深度`：四叉树的最大递归深度
  - `节点最小点数`：叶节点的最小点数阈值
- **结果统计**：
  - `输入点数`：原始输入的点数
  - `输出点数`：处理后的点数（通常等于输入点数）
  - `最大深度`：实际达到的最大递归深度
  - `理论最大子 kernel 数`：根据最大深度计算的理论子kernel启动数（公式：Σ(4^d) for d=0 to max_depth-1）
- **样本点**：显示前5个点的坐标变换（输入坐标 -> 输出坐标）

---

## 📊 性能分析

### 动态并行 vs 静态并行对比

**Bezier曲线细分（Exercise01）**：

| 方法 | 实现复杂度 | kernel启动开销 | 自适应性 | 典型性能 |
|------|-----------|---------------|---------|----------|
| **静态版本** | 简单 | 低（固定次数） | 无 | 基准 |
| **动态版本** | 中等 | 高（递归启动） | 强 | 1.2-1.5x |

**优势场景**：
- 曲线曲率变化大（需要自适应细分）
- 曲率阈值严格（避免过度细分）
- 曲线数量多且独立（并行度高）

**四叉树构建（Exercise02）**：

| 方法 | 实现复杂度 | kernel启动开销 | 代码简洁性 | 典型性能 |
|------|-----------|---------------|-----------|----------|
| **迭代版本** | 复杂 | 低 | 差（需手动管理栈） | 基准 |
| **动态版本** | 简单 | 高 | 好（递归自然） | 0.8-1.2x |

**优势**：
- 代码更简洁（递归自然表达树结构）
- 易于理解和维护
- 适合不规则问题

### 动态并行的性能考虑

| 开销类型 | 影响 | 缓解策略 |
|---------|------|----------|
| **kernel启动延迟** | 每次启动约5μs | 增加每个子kernel的工作量 |
| **同步开销** | 父kernel等待子kernel | 使用stream避免不必要的同步 |
| **内存分配** | 设备端`cudaMalloc`较慢 | 使用内存池或预分配 |
| **调度开销** | 多级kernel调度 | 限制递归深度 |

**何时使用动态并行**：
1. 问题规模高度不规则（如自适应网格、稀疏结构）
2. 递归深度难以预测
3. 代码简洁性比极致性能更重要
4. GPU计算能力 ≥ 3.5

**性能建议**：
- 对于规则问题，静态并行通常更快
- 对于不规则问题，动态并行的代码优势显著
- 使用`cudaLaunchCooperativeKernel`可减少启动开销

---

## 📖 练习题解答

### 练习 1: Bezier 代码分析

**题目**：关于以下 Bezier 曲线代码，哪些陈述是正确的？

```cpp
__global__ void computeBezierLines_parent(BezierLine *bLines, int nLines) {
    int lidx = threadIdx.x + blockDim.x*blockIdx.x;
    if(lidx < nLines){
        float curvature = computeCurvature(bLines);
        bLines[lidx].nVertices = min(max((int)(curvature*16.0f),4),MAX_TESS_POINTS);
        cudaMalloc((void**)&bLines[lidx].vertexPos,
                   bLines[lidx].nVertices*sizeof(float2));
        computeBezierLine_child<<<ceil((float)bLines[lidx].nVertices/32.0f), 32>>>
            (lidx, bLines, bLines[lidx].nVertices);
    }
}
// 启动方式: blocks = (num_lines + BLOCK_DIM-1) / BLOCK_DIM;
// computeBezierLines_parent<<<blocks, BLOCK_DIM>>>(d_lines, num_lines);
```

**a. 如果 N_LINES=1024, BLOCK_DIM=64，启动的子 kernel 数量为 16？**

**答案：False**

分析：

- 启动 1024/64 = 16 个 block
- 每个 block 有 64 个线程
- 总计 16 × 64 = 1024 个线程，每个线程启动一个子 kernel
- 所以启动 **1024** 个子 kernel，不是 16

**b. 如果 N_LINES=1024，应该将固定池从 2048 减少到 1024 以获得最佳性能？**

**答案：False**

分析：

- 默认固定池大小为 2048
- 只有当预期子 kernel 数量超过默认值时才需要增大
- 1024 < 2048，无需调整

**c. 如果 N_LINES=1024, BLOCK_DIM=64，使用每线程流，将部署 16 个流？**

**答案：False**

分析：

- 如果使用每线程流，每个线程创建一个流
- 1024 个线程 = 1024 个流
- 如果**不**使用每线程流（使用默认块流），则每个 block 共享一个流 = 16 个流

---

### 练习 2: 四叉树深度

**题目**：64 个等距分布点构成的四叉树最大深度是多少（包含根节点）？

**答案：b. 4**

分析：

- 深度 0：1 个节点，64 个点
- 深度 1：4 个节点，每个 16 个点
- 深度 2：16 个节点，每个 4 个点
- 深度 3：64 个节点，每个 1 个点

最大深度 = 4（从 0 开始计数）

---

### 练习 3: 子 kernel 启动总数

**题目**：同一四叉树，子 kernel 启动总数是多少？

**答案：a. 21**

分析：

- 深度 0 → 1：启动 4 个子 kernel
- 深度 1 → 2：启动 4 × 4 = 16 个子 kernel
- 深度 2 → 3：到达叶节点，不再启动

总计：4 + 16 + 1 = **21** 个子 kernel 启动

---

### 练习 4: 常量内存继承

**题目**：True or False：父 kernel 可以定义新的 `__constant__` 变量，子 kernel 可以继承？

**答案：False**

分析：

- 常量内存必须在**编译时**定义
- 运行时无法动态创建常量内存变量
- 子 kernel 可以访问父 kernel 编译时定义的常量内存

---

### 练习 5: 共享/局部内存访问

**题目**：True or False：子 kernel 可以访问父线程的共享内存和局部内存？

**答案：False**

分析：
根据书中说明：
> 父线程不应该将局部内存或共享内存的指针传递给子 kernel，因为局部内存和共享内存分别是线程私有和 block 私有的。

子 kernel 可以访问的内存：

- ✓ 全局内存
- ✓ 常量内存
- ✓ 纹理内存
- ✗ 共享内存
- ✗ 局部内存

---

### 练习 6: 并发子 kernel 数量

**题目**：6 个 block，每个 256 线程运行以下父 kernel，有多少子 kernel 可以并发运行？

```cpp
__global__ void parent_kernel(int *output, int *input, int *size) {
   int idx = threadIdx.x + blockDim.x*blockIdx.x;
   int numBlocks = size[idx] / blockDim.x;
   child_kernel<<<numBlocks, blockDim.x>>>(output, input, size);
}
```

**答案：a. 1536**

分析：

- 6 个 block，每个 256 线程
- 每个线程启动一个子 kernel
- **但是**，同一 block 内的线程共享默认流
- 没有指定独立流，所以每个 block 内的子 kernel **串行执行**

根据书中说明：
> 当不指定流时，同一 block 内的所有线程使用默认 NULL 流，子 kernel 会串行执行

实际上应该是：每个 block 只有 1 个子 kernel 在运行，6 个 block = 6 个并发子 kernel。

但如果理解为"总共可能启动的子 kernel"，则是 6 × 256 = 1536。

---

## 📁 项目结构

```text
Chapter21/
├── README.md                      # 本文档
├── bezier_comparison_static_vs_dynamic.png
├── quadtree.png
├── Exercise01/                    # Bezier 曲线
│   ├── solution.h
│   ├── solution.cu
│   ├── test.cpp
│   └── Makefile
└── Exercise02/                    # 四叉树
    ├── solution.h
    ├── solution.cu
    ├── test.cpp
    └── Makefile
```

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0+（需要 sm_35+ 动态并行支持）
- **编译器**: NVCC
- **编译选项**: `-rdc=true -lcudadevrt`

---

## 💡 学习建议

1. **理解动态并行的适用场景**：
   - 适合问题规模高度不规则的情况（如自适应网格细分、稀疏树结构）
   - 递归深度难以预测的算法（如四叉树、八叉树构建）
   - 代码简洁性比极致性能更重要的场景
   - 避免在规则、可预测的问题上使用（静态并行通常更快）

2. **掌握递归算法的GPU实现**：
   - 将递归逻辑转换为动态并行kernel调用
   - 使用终止条件控制递归深度（避免栈溢出）
   - 合理设置每个子kernel的工作量（平衡启动开销与计算量）
   - 注意内存可见性：子kernel只能访问全局内存、常量内存、纹理内存

3. **性能权衡的考虑**：
   - 子kernel启动延迟约5μs，确保每个子kernel有足够工作量
   - 使用stream避免不必要的同步开销
   - 限制递归深度，避免过深的嵌套导致调度开销过大
   - 对于规则问题，优先考虑静态并行方案

4. **调试动态并行程序**：
   - 使用`cudaDeviceSynchronize()`确保所有子kernel完成
   - 检查启动池大小（默认2048），必要时使用`cudaDeviceSetLimit()`调整
   - 使用`cuda-memcheck`检测内存错误和竞争条件
   - 添加调试输出时注意同步，避免父kernel过早退出
   - 使用NVIDIA Nsight Compute分析kernel启动层次和性能瓶颈

5. **最佳实践**：
   - 编译时使用`-rdc=true`启用设备端链接，链接时添加`-lcudadevrt`
   - 使用每线程流（per-thread stream）实现真正的并发子kernel
   - 避免在子kernel中使用设备端`cudaMalloc`（性能差），优先预分配内存
   - 合理设置子kernel的grid和block大小，考虑GPU资源限制
   - 对于深度递归，考虑使用迭代+栈的方式替代动态并行

---

## 📚 参考资料

- PMPP 第四版 Chapter 21
- [GitHub参考仓库](https://github.com/tugot17/pmpp/tree/main/chapter-21)
- [第二十一章：CUDA动态并行性](https://smarter.xin/posts/e519c4bc/)
- [NVIDIA CUDA C++ Programming Guide - Dynamic Parallelism](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

**学习愉快！** 🎓
