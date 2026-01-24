#ifndef SOLUTION_H
#define SOLUTION_H

#include <cuda_runtime.h>

/**
 * 第十二章：归并 - 并行归并实现
 * 
 * 包含多种归并实现：
 * 1. 顺序归并（CPU参考）
 * 2. 基础并行归并（图12.9）
 * 3. 分块归并（图12.11-12.13）
 * 
 * 核心算法：
 * - co_rank: 二分搜索找到归并位置
 * - merge_sequential: 顺序归并两个有序数组
 */

// 分块大小
#define TILE_SIZE 256

// ====================== 核心函数 ======================

/**
 * 协同排名函数（co-rank）
 * 找到合并结果中第 k 个元素来自 A 的位置
 */
__host__ __device__ int co_rank(int k, float* A, int m, float* B, int n);

/**
 * 顺序归并（CPU/GPU）
 */
void merge_sequential(float* A, int m, float* B, int n, float* C);

// ====================== GPU 归并函数 ======================

/**
 * 基础并行归并（图12.9）
 * 每个线程独立计算 co-rank 并处理一段
 */
void merge_basic_gpu(float* A, int m, float* B, int n, float* C);

/**
 * 分块并行归并（图12.11-12.13）
 * 使用共享内存优化，减少全局内存 co-rank 调用
 */
void merge_tiled_gpu(float* A, int m, float* B, int n, float* C);

#endif // SOLUTION_H
