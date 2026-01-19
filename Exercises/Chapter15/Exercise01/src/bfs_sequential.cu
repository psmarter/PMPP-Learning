/**
 * 第十五章：图遍历 - 串行BFS实现
 * 
 * 参考：chapter-15/code/src/bfs_sequential.cu
 * 
 * 提供标准的CPU BFS实现，用于验证GPU实现的正确性
 */

#include "../include/bfs_sequential.h"
#include "../include/graph_structures.h"
#include <cstdio>
#include <cstdlib>
#include <queue>

/**
 * 串行广度优先搜索（CPU实现）
 * 
 * 算法步骤：
 * 1. 初始化所有顶点的level为-1（未访问）
 * 2. 将起点加入队列，level设为0
 * 3. 当队列非空时：
 *    a. 取出队首顶点
 *    b. 遍历其所有邻居
 *    c. 未访问的邻居标记level并加入队列
 * 
 * @param graph CSR格式的图
 * @param startingNode 起始顶点
 * @return level数组（需要调用者释放）
 */
int* bfs(const CSRGraph& graph, int startingNode) {
    // 分配level数组
    int* levels = (int*)malloc(sizeof(int) * graph.numVertices);
    
    // 初始化所有顶点为未访问（-1）
    for (int i = 0; i < graph.numVertices; i++) {
        levels[i] = -1;
    }
    
    // 标准队列
    std::queue<int> q;
    
    // 起点level为0
    levels[startingNode] = 0;
    q.push(startingNode);
    
    // BFS主循环
    while (!q.empty()) {
        int vertex = q.front();
        q.pop();
        
        // 遍历当前顶点的所有邻居
        for (int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
            int neighbour = graph.dst[edge];
            
            // 如果邻居未访问
            if (levels[neighbour] == -1) {
                levels[neighbour] = levels[vertex] + 1;
                q.push(neighbour);
            }
        }
    }
    
    return levels;
}
