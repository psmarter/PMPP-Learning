#ifndef BFS_PARALLEL_H
#define BFS_PARALLEL_H

/**
 * 第十五章：图遍历 - 并行BFS
 * 
 * 实现6种并行BFS算法：
 * 1. Push Vertex-Centric BFS
 * 2. Pull Vertex-Centric BFS
 * 3. Edge-Centric BFS
 * 4. Frontier Vertex-Centric BFS（基础版）
 * 5. Frontier Vertex-Centric BFS（带私有化优化）
 * 6. Direction-Optimized BFS（Push/Pull动态切换）
 */

#include "graph_structures.h"

// ====================== Push Vertex-Centric BFS ======================

/**
 * Push 模式 BFS（顶点中心）
 * 
 * 每个线程处理一个顶点
 * 如果顶点在当前层，则遍历其所有邻居并更新
 * 
 * @param deviceGraph GPU上的CSR图
 * @param startingNode 起始顶点
 * @return level数组（需要调用者释放）
 */
int* bfsParallelPushVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode);

// ====================== Pull Vertex-Centric BFS ======================

/**
 * Pull 模式 BFS（顶点中心）
 * 
 * 每个线程处理一个未访问顶点
 * 检查其邻居是否在上一层，如果是则更新自己
 * 
 * @param deviceGraph GPU上的CSC图
 * @param startingNode 起始顶点
 * @return level数组（需要调用者释放）
 */
int* bfsParallelPullVertexCentricDevice(const CSCGraph& deviceGraph, int startingNode);

// ====================== Edge-Centric BFS ======================

/**
 * Edge-Centric BFS
 * 
 * 每个线程处理一条边
 * 如果源顶点在当前层，尝试更新目标顶点
 * 
 * @param deviceGraph GPU上的COO图
 * @param startingNode 起始顶点
 * @return level数组（需要调用者释放）
 */
int* bfsParallelEdgeCentricDevice(const COOGraph& deviceGraph, int startingNode);

// ====================== Frontier Vertex-Centric BFS ======================

/**
 * Frontier 模式 BFS（基础版）
 * 
 * 使用稀疏前沿队列，只处理前沿中的顶点
 * 使用全局原子操作管理队列
 * 
 * @param deviceGraph GPU上的CSR图
 * @param startingNode 起始顶点
 * @return level数组（需要调用者释放）
 */
int* bfsParallelFrontierVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode);

/**
 * Frontier 模式 BFS（带私有化优化）
 * 
 * 使用共享内存私有化技术减少全局原子操作
 * Block级别的局部队列
 * 
 * @param deviceGraph GPU上的CSR图
 * @param startingNode 起始顶点
 * @return level数组（需要调用者释放）
 */
int* bfsParallelFrontierVertexCentricOptimizedDevice(const CSRGraph& deviceGraph, int startingNode);

// ====================== Direction-Optimized BFS ======================

/**
 * Direction-Optimized BFS（方向优化）
 * 
 * 根据前沿大小动态切换Push/Pull策略
 * - 前沿小时用Push
 * - 前沿大时用Pull
 * 
 * @param deviceCSRGraph GPU上的CSR图（用于Push）
 * @param deviceCSCGraph GPU上的CSC图（用于Pull）
 * @param startingNode 起始顶点
 * @param alpha 切换阈值（访问顶点比例）
 * @return level数组（需要调用者释放）
 */
int* bfsDirectionOptimizedDevice(const CSRGraph& deviceCSRGraph, const CSCGraph& deviceCSCGraph, 
                                 int startingNode, float alpha);

#endif // BFS_PARALLEL_H
