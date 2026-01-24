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

**相关博客笔记**：[第十五章：图遍历](https://smarter.xin/posts/70b05668/)

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
| `bfsParallelSingleBlockDevice` | 15.7 | 单块BFS：共享内存队列（练习3） |

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
7. Single-Block BFS (Exercise 3)... ✅ 结果正确！

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
Single-Block BFS: 0.50 ms (9.42x speedup)
```

---

## 📖 练习题解答

### 练习 1: 手动BFS遍历

**题目：** 考虑书中图15.1的有向图，手动执行不同BFS实现。

**图的表示（基于参考Trace推导的图拓扑）：**

**邻接矩阵**（8×8）:

```
  0 1 2 3 4 5 6 7
0 [0 0 1 0 0 1 0 0]
1 [1 0 0 0 1 0 0 0]
2 [1 0 0 1 0 0 0 0]
3 [1 0 0 0 0 0 1 0]
4 [0 1 0 0 0 0 0 0]
5 [1 1 0 0 0 0 0 1]
6 [0 0 0 1 0 0 0 0]
7 [0 0 0 0 1 0 1 0]
```

**CSR 表示**:

```
srcPtrs = [0, 2, 4, 6, 8, 9, 12, 13, 15]
dst     = [2, 5, 0, 4, 0, 3, 0, 6, 1, 0, 1, 7, 3, 4, 6]
```

**i. Vertex-centric Push BFS:**

从顶点0出发，`BLOCK_SIZE = 256`。

- **Iteration 1 (Level 1):**
  - **启动线程**: 8个（覆盖所有顶点）
  - **遍历邻居的线程**: 1个（顶点0，Level 0）
  - **新访问顶点**: {2, 5}
  - **更新**: level[2]=1, level[5]=1

- **Iteration 2 (Level 2):**
  - **启动线程**: 8个
  - **遍历邻居的线程**: 2个（顶点2, 5）
  - **新访问顶点**: {1, 3, 7}
  - **更新**: level[1]=2, level[3]=2, level[7]=2

- **Iteration 3 (Level 3):**
  - **启动线程**: 8个
  - **遍历邻居的线程**: 3个（顶点1, 3, 7）
  - **新访问顶点**: {4, 6}
  - **更新**: level[4]=3, level[6]=3

- **Iteration 4 (Level 4):**
  - **启动线程**: 8个
  - **遍历邻居的线程**: 2个（顶点4, 6）
  - **新访问顶点**: 无（邻居都已访问）
  - **终止条件**: hostNewVertexVisited = 0

**总迭代次数：4次**（参考仓库可能记为5次如果包含最后一次空检查）

**ii. Vertex-centric Pull BFS:**

- **Iteration 1:**
  - 启动8个线程。
  - 7个未访问顶点检查前驱。
  - **标记顶点**: 2, 5 (发现前驱0)

- **Iteration 2:**
  - 启动8个线程。
  - 5个未访问顶点(1,3,4,6,7)检查前驱。
  - **标记顶点**: 1, 3, 7 (发现前驱2或5)

- **Iteration 3:**
  - 启动8个线程。
  - 2个未访问顶点(4,6)检查前驱。
  - **标记顶点**: 4, 6 (发现前驱1,7或3)

- **Iteration 4:**
  - 无新顶点发现。

**iii. Edge-centric BFS:**

总边数 = 15条，`BLOCK_SIZE = 256`。

- **Iteration 1:**
  - 启动15个线程（覆盖所有边）。
  - **有效更新**: 边 0->2, 0->5 更新顶点 2, 5。

- **Iteration 2:**
  - 启动15个线程。
  - **有效更新**: 边 5->1, 5->7, 2->3 更新顶点 1, 7, 3。

- **Iteration 3:**
  - 启动15个线程。
  - **有效更新**: 边 1->4, 7->4, 3->6, 7->6 更新顶点 4, 6。
  - 注意：对顶点4和6有重复更新（"impotent works"）。

- **Iteration 4:**
  - 启动15个线程。
  - 无更新。

**iv. Frontier Vertex-centric Push BFS:**

- **Iteration 1:** 前沿={0}。启动1个线程。生成新前沿={2, 5}。
- **Iteration 2:** 前沿={2, 5}。启动2个线程。生成新前沿={1, 3, 7}。
- **Iteration 3:** 前沿={1, 3, 7}。启动3个线程。生成新前沿={4, 6}。
- **Iteration 4:** 前沿={4, 6}。启动2个线程。无新前沿生成。

### 练习 2: 方向优化 BFS

**题目：** 实现 Section 15.3 中的方向优化 BFS。

**解答：**

代码位置：`Exercise01/src/bfs_parallel.cu` 中的 `bfsDirectionOptimizedDevice()` 函数。

**核心思想：**

根据前沿大小动态选择策略：

- **前沿小**（早期）→ 使用 **Push**（CSR图，遍历邻居）
- **前沿大**（中期）→ 切换到 **Pull**（CSC图，检查前驱）
- 切换条件：`visitedFraction > α`（例如 α = 0.1）

### 练习 3: 单块 BFS (Exercise 3)

**题目：** 实现 Section 15.7 中的单块 BFS kernel。

**解答：**

**代码位置**：`Exercise01/src/bfs_parallel.cu` 中的 `bfsParallelSingleBlockDevice()` 和 `bfs_single_block_kernel`。

该实现使用单个CUDA Block和共享内存来维护前沿队列，当队列大小超过共享内存容量时（或需要扩展到下一层时），逻辑上可以回退到全局队列或简单地作为演示版本仅处理该Block能处理的部分。本实现包含了一个简化的单Block内核。

参考实现逻辑：

```cpp
__global__ void bfs_single_block_kernel(...) {
    __shared__ int localFrontier[LOCAL_CAPACITY];
    // 使用共享内存维护前沿
    // 适合小图或各Level顶点数少的情况
    // 如果溢出共享内存，需回退到全局队列模式
}
```

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

5. **性能分析与调优**：使用 Nsight Systems 分析 BFS 各阶段的耗时，重点关注前沿队列构建和原子操作的开销；对于大规模图，考虑使用 cuGraph 库（如 `cugraph::bfs`），它针对不同图类型和硬件进行了优化；实际应用中，方向优化 BFS 通常能获得最佳性能

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
- [第十五章：图遍历](https://smarter.xin/posts/70b05668/)

**学习愉快！** 🎓
