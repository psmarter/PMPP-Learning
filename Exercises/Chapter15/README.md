# 第十五章：图遍历

《Programming Massively Parallel Processors》第四版 - 学习笔记与练习

## 📚 学习内容

本章系统梳理图遍历算法及其 GPU 并行化技术：

- 图的表示：CSR、CSC、COO 格式
- 广度优先搜索（BFS）的串行和并行实现
- Push vs Pull 策略（Section 15.2-15.3）
- Frontier 队列管理（Section 15.4）
- 私有化优化技术（Section 15.5）
- 方向优化：动态切换策略（Section 15.3）
- 负载平衡与性能优化

**相关博客笔记**：[PMPP-第十五章：图遍历.md](../../Blogs/PMPP-第十五章：图遍历.md)

---

## 💻 代码实现

### Exercise01 - BFS 完整实现

实现6种 BFS 算法，对应书中不同优化策略。

**代码位置**：`Exercise01/`

**文件结构**：

```
Exercise01/
├── include/          # 头文件目录
│   ├── bfs_parallel.h
│   ├── bfs_sequential.h
│   ├── device_memory.h
│   ├── graph_conversions.h
│   ├── graph_generators.h
│   ├── graph_structures.h
│   └── utils.h
├── src/              # 实现文件目录
│   ├── bfs_parallel.cu
│   ├── bfs_sequential.cu
│   ├── device_memory.cu
│   ├── graph_conversions.cu
│   ├── graph_generators.cu
│   ├── main.cu
│   └── utils.cu
└── Makefile
```

**实现列表**：

| 实现 | 书中对应 | 特点 |
| ---- | -------- | ---- |
| `bfsParallelPushVertexCentricDevice` | 15.2 | Push模式：从当前层推送 |
| `bfsParallelPullVertexCentricDevice` | 15.3 | Pull模式：从前驱拉取 |
| `bfsParallelEdgeCentricDevice` | 15.2 | 边中心：每线程处理一条边 |
| `bfsParallelFrontierVertexCentricDevice` | 15.4 | Frontier队列：稀疏表示 |
| `bfsParallelFrontierVertexCentricOptimizedDevice` | 15.5 | 私有化：共享内存优化 |
| `bfsDirectionOptimizedDevice` | 15.3 | 方向优化：动态切换（练习2） |

**核心代码**：

```cuda
// Push Vertex-Centric Kernel
__global__ void bsf_push_vertex_centric_kernel(CSRGraph graph, int* levels, 
                                               int* newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < graph.numVertices) {
        if (levels[vertex] == currLevel - 1) {
            // 遍历当前层顶点的所有邻居
            for (unsigned int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = graph.dst[edge];
                if (levels[neighbour] == -1) {
                    levels[neighbour] = currLevel;
                    *newVertexVisited = 1;
                }
            }
        }
    }
}

// Frontier Vertex-Centric Kernel（带私有化）
__global__ void bsf_frontier_vertex_centric_with_privatization_kernel(CSRGraph csrGraph, int* levels, 
                                                                      int* prevFrontier, int* currFrontier, 
                                                                      int numPrevFrontier, int* numCurrFrontier, 
                                                                      int currLevel) {
    // 共享内存私有化：减少全局原子操作
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();
    
    // BFS 主体
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (atomicCAS(&levels[neighbor], -1, currLevel) == -1) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    // 添加到共享内存队列
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    // 共享内存满，写入全局内存
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    // ... 提交阶段
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
  第十五章：图遍历
  Breadth-First Search - Multiple Implementations
================================================================

=== 正确性验证 ===

生成测试图（2000个顶点）...
1. Push Vertex-Centric BFS... ✅ 结果正确！
2. Pull Vertex-Centric BFS... ✅ 结果正确！
3. Edge-Centric BFS... ✅ 结果正确！
4. Frontier BFS (基础版)... ✅ 结果正确！
5. Frontier BFS (优化版)... ✅ 结果正确！
6. Direction-Optimized BFS... ✅ 结果正确！

所有BFS实现通过正确性验证！

=== 性能基准测试 ===

图规模：10000 个顶点
--------------------
生成无标度图...
Sequential BFS: 4.71 ms
Push Vertex-Centric BFS: 1.00 ms (4.71x speedup)
Pull Vertex-Centric BFS: 0.32 ms (14.72x speedup)
Edge-Centric BFS: 0.13 ms (36.23x speedup)
Frontier-based BFS: 1.77 ms (2.66x speedup)
Optimized Frontier-based BFS: 1.83 ms (2.57x speedup)
Direction-Optimized BFS: 0.35 ms (13.46x speedup)
```

---

## 📖 练习题解答

### 练习 1: 手动BFS遍历

**题目：** 考虑书中图15.1的有向图，手动执行不同BFS实现。

**图的表示：**

**邻接矩阵**（8×8）:

```
  0 1 2 3 4 5 6 7
0 [0 1 1 0 0 0 0 0]
1 [0 0 0 1 1 0 0 0]
2 [0 0 0 0 1 0 0 0]
3 [0 0 0 0 0 1 1 0]
4 [0 0 0 0 0 0 1 0]
5 [0 0 0 0 0 0 0 1]
6 [0 0 0 0 0 0 0 1]
7 [0 0 0 0 0 0 0 0]
```

**CSR 表示**:

```
srcPtrs = [0, 2, 4, 5, 7, 8, 9, 10, 10]
dst     = [1, 2, 3, 4, 4, 5, 6, 6, 7, 7]
```

**i. Vertex-centric Push BFS:**

从顶点0出发，`BLOCK_SIZE = 256`。

- **Iteration 1, currLevel = 1:**
  - 线程启动：⌈8/256⌉×256 = 256 个线程
  - 活跃线程：1个（顶点0在level 0）
  - 顶点被访问：{1, 2}
  - 更新：level[1] = 1, level[2] = 1

- **Iteration 2, currLevel = 2:**
  - 线程启动：256个线程
  - 活跃线程：2个（顶点1,2在level 1）
  - 顶点被访问：{3, 4}
  - 更新：level[3] = 2, level[4] = 2

- **Iteration 3, currLevel = 3:**
  - 线程启动：256个线程
  - 活跃线程：2个（顶点3,4在level 2）
  - 顶点被访问：{5, 6}
  - 更新：level[5] = 3, level[6] = 3

- **Iteration 4, currLevel = 4:**
  - 线程启动：256个线程
  - 活跃线程：2个（顶点5,6在level 3）
  - 顶点被访问：{7}
  - 更新：level[7] = 4

- **Iteration 5, currLevel = 5:**
  - 线程启动：256个线程
  - 活跃线程：0个
  - 终止

**总迭代次数：5次**，**总线程启动：256×5 = 1280个**

**ii. Vertex-centric Pull BFS:**

- 每次迭代启动256个线程（所有顶点）
- 检查未访问顶点的前驱
- **总迭代次数：5次**，**总线程启动：1280个**

**iii. Edge-centric BFS:**

总边数 = 10条，`BLOCK_SIZE = 256`。

- 每次迭代启动 ⌈10/256⌉×256 = 256 个线程
- **总迭代次数：5次**，**总线程启动：1280个**

**iv. Frontier Vertex-centric Push BFS:**

- **Iteration 1:** 前沿 = {0}, 启动256个线程，访问{1,2}
- **Iteration 2:** 前沿 = {1,2}, 启动256个线程，访问{3,4}
- **Iteration 3:** 前沿 = {3,4}, 启动256个线程，访问{5,6}
- **Iteration 4:** 前沿 = {5,6}, 启动256个线程，访问{7}
- **Iteration 5:** 前沿 = {7}, 启动256个线程，无新访问

**总迭代次数：5次**，**总线程启动：1280个**

### 练习 2: 方向优化 BFS

**题目：** 实现 Section 15.3 中的方向优化 BFS。

**解答：**

代码位置：`Exercise01/src/bfs_parallel.cu` 中的 `bfsDirectionOptimizedDevice()` 函数。

**核心思想：**

根据前沿大小动态选择策略：

- **前沿小**（早期）→ 使用 **Push**（CSR图，遍历邻居）
- **前沿大**（中期）→ 切换到 **Pull**（CSC图，检查前驱）
- 切换条件：`visitedFraction > α`（例如 α = 0.1）

**算法流程：**

```cpp
int* bfsDirectionOptimizedDevice(const CSRGraph& deviceCSRGraph, 
                                 const CSCGraph& deviceCSCGraph, 
                                 int startingNode, float alpha) {
    // 初始化
    bool usingPush = true;
    int visitedVertices = 1;
    int totalVertices = deviceCSRGraph.numVertices;
    
    while (有新顶点被访问) {
        // 计算访问顶点比例
        float visitedFraction = (float)visitedVertices / totalVertices;
        
        // 动态切换策略
        if (usingPush && visitedFraction > alpha) {
            usingPush = false;  // 切换到Pull
        }
        
        if (usingPush) {
            // 使用Push Kernel（CSR图）
            bsf_push_vertex_centric_kernel<<<...>>>(deviceCSRGraph, ...);
        } else {
           // 使用Pull Kernel（CSC图）
            bsf_pull_vertex_centric_kernel<<<...>>>(deviceCSCGraph, ...);
        }
        
        // 更新visitedVertices
        visitedVertices = countVisitedVertices();
        currLevel++;
    }
    
    return levels;
}
```

**性能优势：**

- 无标度图：通常有 10-20x 加速
- 小世界图：避免中期的大量无效线程
- 自适应：不依赖人工调优

### 练习 3: 单块 BFS（未在本实现中包含）

**题目：** 实现 Section 15.7 中的单块 BFS kernel。

**概念说明：**

单块BFS在共享内存中维护前沿队列，适用于：

- 前沿队列较小的图
- BFS的前几层迭代
- 与多块模式混合使用

**优化点：**

1. 前沿在共享内存中 → 减少全局内存访问
2. 单个block处理 → 避免多块同步开销
3. 溢出处理 → 超出容量时切换到全局队列

**性能权衡：**

- 优势：低延迟、高带宽
- 劣势：受限于共享内存大小（通常48KB）

---

## 🔧 开发环境

- **CUDA Toolkit**: 11.0+
- **编译器**: GCC 7.5+ / Visual Studio 2019+ + NVCC
- **GPU**: NVIDIA 显卡（计算能力 3.5+）

---

## 💡 学习建议

1. **理解图的存储格式**：
   - CSR 适合 Push（遍历出边）
   - CSC 适合 Pull（查找入边）
   - COO 简单但空间效率低

2. **掌握 Push vs Pull**：
   - Push：适合前沿小时（减少线程数）
   - Pull：适合前沿大时（减少写冲突）

3. **优化队列管理**：
   - 使用前沿队列减少无效工作
   - 私有化减少原子操作争用

4. **负载均衡**：
   - 度数差异大的图（无标度）需要动态分配
   - 使用 Work-stealing 或 Dynamic parallelism

---

## 🚀 下一步

完成本章学习后，可以探索：

- 深度优先搜索（DFS）的并行化
- 最短路径算法（Dijkstra、Bellman-Ford）
- PageRank 和其他图算法
- 图神经网络（GNN）的GPU加速

---

## 📚 参考资料

- PMPP 第四版 Chapter 15
- [GitHub参考仓库](https://github.com/tugot17/pmpp/tree/main/chapter-15)
- [PMPP-第十五章：图遍历.md](../../Blogs/PMPP-第十五章：图遍历.md)
