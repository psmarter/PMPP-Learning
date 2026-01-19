#ifndef DEVICE_MEMORY_H
#define DEVICE_MEMORY_H

/**
 * 第十五章：图遍历 - GPU内存管理
 * 
 * 提供图数据结构在GPU上的分配和释放函数
 */

#include "graph_structures.h"

// ====================== CSR图内存管理 ======================

/**
 * 在GPU上分配CSR图
 * @param hostGraph 主机端CSR图
 * @return GPU上的CSR图
 */
CSRGraph allocateCSRGraphOnDevice(const CSRGraph& hostGraph);

/**
 * 释放GPU上的CSR图
 * @param deviceGraph GPU上的CSR图指针
 */
void freeCSRGraphOnDevice(CSRGraph* deviceGraph);

// ====================== CSC图内存管理 ======================

/**
 * 在GPU上分配CSC图
 * @param hostGraph 主机端CSC图
 * @return GPU上的CSC图
 */
CSCGraph allocateCSCGraphOnDevice(const CSCGraph& hostGraph);

/**
 * 释放GPU上的CSC图
 * @param deviceGraph GPU上的CSC图指针
 */
void freeCSCGraphOnDevice(CSC Graph* deviceGraph);

// ====================== COO图内存管理 ======================

/**
 * 在GPU上分配COO图
 * @param hostGraph 主机端COO图
 * @return GPU上的COO图
 */
COOGraph allocateCOOGraphOnDevice(const COOGraph& hostGraph);

/**
 * 释放GPU上的COO图
 * @param deviceGraph GPU上的COO图指针
 */
void freeCOOGraphOnDevice(COOGraph* deviceGraph);

// ====================== Level数组管理 ======================

/**
 * 在GPU上分配并初始化level数组
 * @param numVertices 顶点数
 * @param startVertex 起始顶点
 * @return GPU上的level数组
 */
int* allocateAndInitLevelsOnDevice(int numVertices, int startVertex);

/**
 * 将level数组从GPU拷贝回主机
 * @param d_levels GPU上的level数组
 * @param numVertices 顶点数
 * @return 主机端的level数组（需要调用者释放）
 */
int* copyLevelsToHost(int* d_levels, int numVertices);

#endif // DEVICE_MEMORY_H
