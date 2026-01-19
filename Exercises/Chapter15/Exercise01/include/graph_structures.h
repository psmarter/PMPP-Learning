#ifndef GRAPH_STRUCTURES_H
#define GRAPH_STRUCTURES_H

/**
 * 第十五章：图遍历 - 图数据结构
 * 
 * 定义三种图表示格式：
 * 1. CSR (Compressed Sparse Row) - 适合 Push 模式 BFS
 * 2. CSC (Compressed Sparse Column) - 适合 Pull 模式 BFS
 * 3. COO (Coordinate) - 适合 Edge-Centric BFS
 */

// ====================== 图数据结构 ======================

/**
 * CSR 格式：压缩稀疏行格式
 * 适合从源顶点出发遍历邻居（Push 模式）
 */
struct CSRGraph {
    int* srcPtrs;      // 源顶点指针数组 [numVertices + 1]
    int* dst;          // 目标顶点数组 [numEdges]
    int* values;       // 边权值数组 [numEdges]
    int numVertices;   // 顶点数
};

/**
 * CSC 格式：压缩稀疏列格式
 * 适合从目标顶点查找前驱（Pull 模式）
 */
struct CSCGraph {
    int* dstPtrs;      // 目标顶点指针数组 [numVertices + 1]
    int* src;          // 源顶点数组 [numEdges]
    int* values;       // 边权值数组 [numEdges]
    int numVertices;   // 顶点数
};

/**
 * COO 格式：坐标格式
 * 存储每条边的 (源, 目标, 权值)
 * 适合 Edge-Centric BFS
 */
struct COOGraph {
    int* scr;          // 源顶点数组 [numEdges]
    int* dst;          // 目标顶点数组 [numEdges]
    int* values;       // 边权值数组 [numEdges]
    int numEdges;      // 边数
    int numVertices;   // 顶点数
};

#endif // GRAPH_STRUCTURES_H
