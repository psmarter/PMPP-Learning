#ifndef BFS_SEQUENTIAL_H
#define BFS_SEQUENTIAL_H

/**
 * 第十五章：图遍历 - 串行BFS
 * 
 * 提供CPU上的标准BFS实现，用于正确性验证
 */

#include "graph_structures.h"

/**
 * 串行广度优先搜索（CPU实现）
 * 
 * 使用标准队列算法计算从起点到所有可达顶点的最短距离（层数）
 * 
 * @param graph CSR格式的图
 * @param startingNode 起始顶点
 * @return 每个顶点的level数组（未访问的顶点为-1）
 */
int* bfs(const CSRGraph& graph, int startingNode);

#endif // BFS_SEQUENTIAL_H
