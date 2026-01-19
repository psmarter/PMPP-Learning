---
title: PMPP-第十五章：图遍历
date: 2026-01-19 16:12:39
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 图算法
  - BFS
  - CSR格式
categories: 知识分享
cover: /img/PMPP.jpg
---

## 前言

第十四章我们学习了稀疏矩阵，那一章的重点是 SpMV（稀疏矩阵-向量乘法）。其实，**图（Graph）**和稀疏矩阵是一体两面的：图的邻接矩阵通常就是稀疏矩阵。第十五章我们将深入探讨图算法的核心——**图遍历**，特别是**广度优先搜索（BFS）**。BFS 是最短路径、连通分量、最大流等众多图算法的基础，也是 GPU 处理不规则数据结构的典型案例。

> **📦 配套资源**：本系列文章配有完整的 [GitHub 仓库](https://github.com/psmarter/PMPP-Learning)，包含每章的练习题解答、CUDA 代码实现和详细注释。所有代码都经过测试，可以直接运行。

## 图的表示

### 从现实世界到图

图由节点（Vertex）和边（Edge）组成。

- **社交网络**：节点是人，边是关注/好友关系。
- **道路网**：节点是路口，边是道路。
- **引用网络**：节点是论文，边是引用关系。

这些图通常是**稀疏**的（每个节点平均只连接少数其他节点）且**无标度**的（少数节点有极多连接，俗称"大V"）。

### 存储格式：CSR

上一章介绍的 **CSR (Compressed Sparse Row)** 格式不仅适合 SpMV，也是存储图的标准格式。

对于一个有 $V$ 个节点、$E$ 条边的图：

- **row_ptr** (长度 $V+1$)：`row_ptr[i]` 指向节点 $i$ 的邻居列表在 `col_idx` 中的起始位置。
- **col_idx** (长度 $E$)：存储所有边的目标节点。

```
节点 0 的邻居：col_idx[row_ptr[0]] ... col_idx[row_ptr[1]-1]
节点 i 的邻居数：row_ptr[i+1] - row_ptr[i]
```

## 广度优先搜索 (BFS)

### 问题定义

给定起点 $S$，BFS 需要访问所有可达节点，并计算从 $S$ 到每个节点的最短距离（层数）。

### 串行 BFS

经典实现使用**队列（Queue）**：

```cpp
void BFS_sequential(int S, int *row_ptr, int *col_idx, int *level, int num_nodes) {
    std::queue<int> q;
    q.push(S);
    level[S] = 0; // 初始化其他为 -1

    while (!q.empty()) {
        int u = q.front(); q.pop();
        
        // 遍历所有邻居 v
        for (int i = row_ptr[u]; i < row_ptr[u+1]; i++) {
            int v = col_idx[i];
            
            // 如果 v 未被访问
            if (level[v] == -1) {
                level[v] = level[u] + 1;
                q.push(v);
            }
        }
    }
}
```

### 并行化的挑战

1. **不规则访问**：每个节点的邻居数量差异巨大（负载不均衡）。
2. **动态工作集**：每一层的节点数都在变化。
3. **并发冲突**：多个节点可能同时尝试访问同一个邻居。

## 并行 BFS：按层同步

GPU 适合解决可以分层的任务。BFS 天然分层：
第 0 层（起点） -> 第 1 层邻居 -> 第 2 层邻居 -> ...

我们可以采用**按层同步（Level Synchronous）**的方法：

1. 维护一个**前沿（Frontier）**数组，标记当前层需要扩展的节点。
2. 并行处理 Frontier 中的每个节点，找到下一层的邻居。
3. 更新 Frontier，进入下一轮迭代。

### 初始版本：布尔前沿数组

用一个布尔数组 `F` 表示当前层节点，`next_F` 表示下一层。

```cuda
__global__ void BFS_kernel(int *row_ptr, int *col_idx, int *level, 
                           bool *F, bool *next_F, int num_nodes, int current_depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_nodes && F[tid]) { // 如果我是当前层节点
        F[tid] = false; // 移除自己
        
        // 遍历邻居
        int start = row_ptr[tid];
        int end = row_ptr[tid+1];
        for (int i = start; i < end; i++) {
            int neighbor = col_idx[i];
            
            // 如果邻居未访问
            if (level[neighbor] == -1) {
                level[neighbor] = current_depth + 1;
                next_F[neighbor] = true; // 加入下一层
            }
        }
    }
}
```

**Host 端循环**：

```cpp
while (frontier_not_empty) {
    BFS_kernel<<<...>>>(..., depth);
    cudaDeviceSynchronize();
    swap(F, next_F);
    depth++;
}
```

### 痛点：并发写冲突

```cuda
if (level[neighbor] == -1) {
    level[neighbor] = current_depth + 1; // 冲突！
    next_F[neighbor] = true;
}
```

当多个节点指向同一个邻居时，会发生**写竞争**。
虽然在这里，只要有一个线程写入成功即可（因为距离都是 `current_depth + 1`），且 `next_F` 设为 true 多次也无妨，但在有些硬件或算法变体上，这会导致数据未定义行为。

**更安全的做法**是使用原子操作，但这很慢。
在 BFS 这种距离只更新一次的场景，可以利用**良性竞争**：多线程同时写入相同的值通常是可以接受的，或者使用 `atomicCAS` 保证只有第一个访问者能更新。

```cuda
if (level[neighbor] == -1) {
    // 利用 level 值作为锁
    // 只有原来的值为 -1 时才更新
    // 实际上对于简单的 BFS，直接写在某些架构上可行，但标准做法应考虑数据竞争
}
```

## 优化一：工作效率与稀疏前沿

上面的布尔数组方法有一个大问题：**工作效率极低**。
每一层都要启动 `num_nodes` 个线程，即使当前层只有 1 个节点！

**稀疏前沿（Sparse Frontier）**：
不存储布尔值，而是将当前层节点的 ID 存储在一个紧凑的队列中。
`Queue: [0, 5, 12]`

这样只需启动 `num_frontier` 个线程。这需要**动态队列管理**。

### 全局原子队列

```cuda
__global__ void BFS_queue_kernel(..., int *queue, int *q_count, int *next_queue, int *next_q_count) {
    int tid = ...;
    // 从 queue 中取节点 u
    int u = queue[tid]; 
    
    // 遍历邻居 v
    // ...
    if (atomicCAS(&level[v], -1, depth+1) == -1) { // 只有第一个更新成功的才入队
         int pos = atomicAdd(next_q_count, 1);
         next_queue[pos] = v;
    }
}
```

**问题**：所有线程争抢 `next_q_count` 这个全局计数器，原子操作争用极其严重！

## 优化二：私有化与分层队列

类似于第九章的直方图优化，我们可以使用**私有化（Privatization）**技术。

1. **Block 级队列**：每个 Block 在共享内存中维护一个小的局部队列。
2. **局部聚合**：线程发现新邻居时，先写入共享内存队列。
3. **全局提交**：Block 满或结束时，一次性申请全局队列空间，将局部队列拷贝过去。

```cuda
__shared__ int s_queue[BLOCK_SIZE];
__shared__ int s_tail;

// 1. 发现邻居入局部队
int s_pos = atomicAdd(&s_tail, 1);
if (s_pos < BLOCK_SIZE) s_queue[s_pos] = v;

__syncthreads();

// 2. Block 申请全局空间
__shared__ int g_offset;
if (threadIdx.x == 0) {
    g_offset = atomicAdd(global_counter, s_tail);
}
__syncthreads();

// 3. 拷贝到全局
if (threadIdx.x < s_tail) {
    next_queue[g_offset + threadIdx.x] = s_queue[threadIdx.x];
}
```

这大大减少了全局原子操作的次数。

## 优化三：方向优化（Top-Down vs Bottom-Up）

这是 BFS 优化中最重要的策略之一，源自 Scott Beamer 的开创性工作。

### Top-Down (Push)

传统的 BFS 是**推（Push）**模式：

- 看着 frontier 里的节点。
- 检查它们的邻居。
- 如果邻居没访问过，更新邻居。

**优点**：前沿（Frontier）很小时效率高。
**缺点**：前沿很大时，边数可能是节点数的几十倍，大量冗余检查（很多邻居已经被访问过了）。

### Bottom-Up (Pull)

反过来思考：**拉（Pull）**模式。

- 看着所有**未访问**的节点。
- 检查它们的邻居是否在 frontier 里。
- 只要发现**一个**邻居在 frontier 里，我就"被发现"了，更新自己，不需要检查其他邻居。

**优点**：当前沿很大时，不需要遍历所有边，只需找到一个父节点即可停止。
**缺点**：需要扫描所有未访问节点，前沿小时效率极低。

### 混合策略 (Direction-Optimizing BFS)

- 开始时（前沿小）：用 Top-Down。
- 中间时（前沿爆炸）：切换到 Bottom-Up。
- 结尾时（前沿收缩）：切回 Top-Down。

这通常能带来数倍的性能提升。

## 负载均衡：处理无标度网络

在社交网络中，某些节点可能有百万级邻居（如名人），而大多节点只有几个邻居。
如果一个线程处理一个节点：

- 线程 A 处理 2 个邻居（瞬间完成）。
- 线程 B 处理 100万 个邻居（卡死整个 Warp/Block）。

**解决方案**：

1. **节点并行**：每个线程处理一个节点（适合普通节点）。
2. **边并行**：对于度数极大的节点，让整个 Warp 甚至整个 Block 协作处理它的邻居列表。

这需要预处理来识别大度节点，类似 SpMV 中的 CSR-Vector 或 CSR-Stream 策略。

## 总结

第十五章展示了如何用 GPU 解决图遍历问题。相比于规则的矩阵运算，图算法充满了挑战：

1. **CSR 格式**：是图和稀疏矩阵的桥梁。
2. **按层同步**：是并行 BFS 的基本框架。
3. **队列管理**：从简单的布尔数组到复杂的双层队列（私有化），是为了解决工作效率和原子争用问题。
4. **方向优化**：Push vs Pull 的动态切换，体现了算法设计对数据特征的适应。
5. **负载均衡**：应对无标度特性，需要细粒度的任务调度。

图算法是高性能计算（HPC）皇冠上的明珠之一（Graph500 排名），掌握了它，意味着你对 GPU 并行模式的理解达到了新的高度。

---

**参考资料：**

- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- Merrill, D., et al. (2012). *Scalable GPU Graph Traversal*. PPoPP.
- Beamer, S., et al. (2012). *Direction-Optimizing Breadth-First Search*. SC12.

---

> **本文 GitHub 仓库**: [https://github.com/psmarter/PMPP-Learning](https://github.com/psmarter/PMPP-Learning)
