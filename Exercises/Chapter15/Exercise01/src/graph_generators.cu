/**
 * 第十五章：图遍历 - 图生成器实现
 * 
 * 参考：chapter-15/code/src/graph_generators.cu
 * 
 * 实现两种图生成算法：
 * 1. Barabási-Albert 无标度图
 * 2. Watts-Strogatz 小世界图
 */

#include "../include/graph_generators.h"
#include "../include/graph_structures.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

/**
 * 生成无标度图（Barabási-Albert 模型）
 * 
 * 算法步骤：
 * 1. 初始化一个小的完全连接图（m0 个顶点）
 * 2. 逐个添加新顶点，使用优先连接（preferential attachment）
 * 3. 新顶点倾向于连接到度数高的现有顶点
 * 
 * @param numVertices 顶点总数
 * @param edgesPerNewVertex 每个新顶点添加的边数（m）
 * @return COO 格式的无标度图
 */
COOGraph generateScaleFreeGraphCOO(int numVertices, int edgesPerNewVertex) {
    // 参数边界检查
    if (numVertices < 3) {
        numVertices = 3;  // 最少3个顶点
    }
    if (edgesPerNewVertex >= numVertices) {
        edgesPerNewVertex = numVertices - 1;
    }
    if (edgesPerNewVertex < 1) {
        edgesPerNewVertex = 1;
    }
    
    // 初始完全图的大小
    int m0 = edgesPerNewVertex + 1;
    if (m0 >= numVertices) {
        m0 = numVertices - 1;
    }
    
    // 计算最大可能边数（无向图，每条边存储两次）
    int maxPossibleEdges = m0 * (m0 - 1) + (numVertices - m0) * edgesPerNewVertex * 2;
    
    // 分配 COO 图内存
    COOGraph graph;
    graph.numVertices = numVertices;
    graph.scr = (int*)malloc(sizeof(int) * maxPossibleEdges);
    graph.dst = (int*)malloc(sizeof(int) * maxPossibleEdges);
    graph.values = (int*)malloc(sizeof(int) * maxPossibleEdges);
    
    // 设置随机种子
    srand(time(NULL));
    
    int edgeIdx = 0;
    
    // 步骤1：创建初始完全图
    for (int i = 0; i < m0; i++) {
        for (int j = 0; j < m0; j++) {
            if (i != j) {
                graph.scr[edgeIdx] = i;
                graph.dst[edgeIdx] = j;
                graph.values[edgeIdx] = 1;
                edgeIdx++;
            }
        }
    }
    
    // 度数数组（用于优先连接）
    int* degree = (int*)calloc(numVertices, sizeof(int));
    
    // 初始化度数
    for (int i = 0; i < edgeIdx; i++) {
        degree[graph.scr[i]]++;
        degree[graph.dst[i]]++;
    }
    
    // 步骤2：逐个添加新顶点
    for (int newVertex = m0; newVertex < numVertices; newVertex++) {
        int edgesAdded = 0;
        
        // 计算度数总和（用于概率选择）
        int totalDegree = 0;
        for (int i = 0; i < newVertex; i++) {
            totalDegree += degree[i];
        }
        
        // 如果度数总和为0，初始化为均匀分布
        if (totalDegree == 0) {
            for (int i = 0; i < newVertex; i++) {
                degree[i] = 1;
            }
            totalDegree = newVertex;
        }
        
        // 追踪已连接的顶点（避免重复）
        bool* connected = (bool*)calloc(newVertex, sizeof(bool));
        
        // 添加 m 条边
        while (edgesAdded < edgesPerNewVertex && edgesAdded < newVertex) {
            // 基于度数选择目标顶点
            int target = -1;
            int randomValue = rand() % totalDegree;
            int cumulativeProbability = 0;
            
            for (int i = 0; i < newVertex; i++) {
                if (connected[i]) {
                    continue;  // 跳过已连接的顶点
                }
                
                cumulativeProbability += degree[i];
                if (randomValue < cumulativeProbability) {
                    target = i;
                    break;
                }
            }
            
            // 如果没有找到目标，随机选择一个未连接的顶点
            if (target == -1) {
                std::vector<int> unconnected;
                for (int i = 0; i < newVertex; i++) {
                    if (!connected[i]) {
                        unconnected.push_back(i);
                    }
                }
                
                if (!unconnected.empty()) {
                    target = unconnected[rand() % unconnected.size()];
                } else {
                    break;  // 没有更多可连接的顶点
                }
            }
            
            // 添加边
            if (target != -1 && !connected[target]) {
                // 边界检查
                if (edgeIdx + 2 > maxPossibleEdges) {
                    printf("错误：边索引超过最大可能边数\n");
                    break;
                }
                
                // 添加正向边
                graph.scr[edgeIdx] = newVertex;
                graph.dst[edgeIdx] = target;
                graph.values[edgeIdx] = 1;
                edgeIdx++;
                
                // 添加反向边（无向图）
                graph.scr[edgeIdx] = target;
                graph.dst[edgeIdx] = newVertex;
                graph.values[edgeIdx] = 1;
                edgeIdx++;
                
                // 更新度数
                degree[newVertex]++;
                degree[target]++;
                connected[target] = true;
                
                edgesAdded++;
            }
        }
        
        free(connected);
    }
    
    // 设置最终边数
    graph.numEdges = edgeIdx;
    
    free(degree);
    return graph;
}

/**
 * 生成小世界图（Watts-Strogatz 模型）
 * 
 * 算法步骤：
 * 1. 初始化环状规则网络（每个顶点连接k/2个最近邻居）
 * 2. 以概率p重连每条边
 * 
 * @param numVertices 顶点数
 * @param k 平均度数（必须是偶数）
 * @param rewireProbability 重连概率（0-1）
 * @return COO 格式的小世界图
 */
COOGraph generateSmallWorldGraphCOO(int numVertices, int k, float rewireProbability) {
    // 参数检查
    if (k % 2 != 0) {
        k--;  // k 必须是偶数
    }
    if (k >= numVertices) {
        k = numVertices - 1;
    }
    if (k < 2) {
        k = 2;
    }
    
    // 无向图的边数
    int totalEdges = numVertices * k / 2;
    
    // 分配 COO 图内存（每条边存储两次）
    COOGraph graph;
    graph.numVertices = numVertices;
    graph.numEdges = totalEdges * 2;
    graph.scr = (int*)malloc(sizeof(int) * totalEdges * 2);
    graph.dst = (int*)malloc(sizeof(int) * totalEdges * 2);
    graph.values = (int*)malloc(sizeof(int) * totalEdges * 2);
    
    // 设置随机种子
    srand(time(NULL));
    
    int edgeIdx = 0;
    
    // 步骤1：创建初始环状网络
    for (int i = 0; i < numVertices; i++) {
        for (int j = 1; j <= k / 2; j++) {
            int neighbor = (i + j) % numVertices;
            
            // 添加正向边
            graph.scr[edgeIdx] = i;
            graph.dst[edgeIdx] = neighbor;
            graph.values[edgeIdx] = 1;
            edgeIdx++;
            
            // 添加反向边
            graph.scr[edgeIdx] = neighbor;
            graph.dst[edgeIdx] = i;
            graph.values[edgeIdx] = 1;
            edgeIdx++;
        }
    }
    
    // 连接矩阵（用于避免重复边）
    bool** connections = (bool**)malloc(sizeof(bool*) * numVertices);
    for (int i = 0; i < numVertices; i++) {
        connections[i] = (bool*)calloc(numVertices, sizeof(bool));
    }
    
    // 初始化连接矩阵
    for (int i = 0; i < edgeIdx; i++) {
        int src = graph.scr[i];
        int dst = graph.dst[i];
        connections[src][dst] = true;
    }
    
    // 步骤2：以概率p重连边（只处理正向边，避免不一致）
    for (int i = 0; i < edgeIdx; i += 2) {
        float random = static_cast<float>(rand()) / RAND_MAX;
        
        if (random < rewireProbability) {
            int src = graph.scr[i];
            int oldDst = graph.dst[i];
            
            // 尝试找到一个新的未连接的目标
            int attempts = 0;
            int newDst = -1;
            bool validTarget = false;
            
            while (!validTarget && attempts < 50) {
                newDst = rand() % numVertices;
                
                // 避免自环和已有连接
                if (newDst != src && !connections[src][newDst]) {
                    validTarget = true;
                }
                
                attempts++;
            }
            
            // 如果找到有效目标，重连边
            if (validTarget) {
                // 移除旧连接
                connections[src][oldDst] = false;
                connections[oldDst][src] = false;
                
                // 添加新连接
                connections[src][newDst] = true;
                connections[newDst][src] = true;
                
                // 更新 COO 图
                graph.dst[i] = newDst;
                
                // 更新反向边
                graph.scr[i + 1] = newDst;
                graph.dst[i + 1] = src;
            }
        }
    }
    
    // 释放内存
    for (int i = 0; i < numVertices; i++) {
        free(connections[i]);
    }
    free(connections);
    
    // 设置最终边数
    graph.numEdges = edgeIdx;
    
    return graph;
}
