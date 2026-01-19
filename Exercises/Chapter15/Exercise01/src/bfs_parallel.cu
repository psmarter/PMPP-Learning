/**
 * 第十五章：图遍历 - 并行BFS实现 (第1部分)
 * 
 * 参考：chapter-15/code/src/bfs_parallel.cu
 * 
 * 实现6种并行BFS算法的kernels和host端代码
 */

#include "../include/bfs_parallel.h"
#include "../include/device_memory.h"
#include "../include/graph_structures.h"
#include "../../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// 常量定义
#define LOCAL_FRONTIER_CAPACITY 256

// ====================== Kernel实现 ======================

/**
 * Kernel: Push Vertex-Centric BFS
 * 
 * 每个线程处理一个顶点
 * 如果顶点在当前层，遍历所有邻居并更新
 */
__global__ void bsf_push_vertex_centric_kernel(CSRGraph graph, int* levels, int* newVertexVisited,
                                               unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < graph.numVertices) {
        if (levels[vertex] == currLevel - 1) {
            // 遍历当前层顶点的所有邻居
            for (unsigned int edge = graph.srcPtrs[vertex]; edge < graph.srcPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = graph.dst[edge];
                // 如果邻居未访问，更新其level
                if (levels[neighbour] == -1) {
                    levels[neighbour] = currLevel;
                    *newVertexVisited = 1;  // 标记有新顶点被访问
                }
            }
        }
    }
}

/**
 * Kernel: Pull Vertex-Centric BFS
 * 
 * 每个线程处理一个未访问顶点
 * 检查其邻居是否在上一层
 */
__global__ void bsf_pull_vertex_centric_kernel(CSCGraph graph, int* levels, int* newVertexVisited,
                                               unsigned int currLevel) {
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vertex < graph.numVertices) {
        if (levels[vertex] == -1) {  // 顶点未访问
            // 检查邻居是否在上一层
            for (unsigned int edge = graph.dstPtrs[vertex]; edge < graph.dstPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = graph.src[edge];
                
                if (levels[neighbour] == currLevel - 1) {
                    levels[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break;  // 找到一个上层邻居即可
                }
            }
        }
    }
}

/**
 * Kernel: Edge-Centric BFS
 * 
 * 每个线程处理一条边
 * 如果源顶点在当前层，尝试更新目标顶点
 */
__global__ void bsf_edge_centric_kernel(COOGraph cooGraph, int* levels, int* newVertexVisited, 
                                        unsigned int currLevel) {
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.scr[edge];
        if (levels[vertex] == currLevel - 1) {
            unsigned int neighbour = cooGraph.dst[edge];
            if (levels[neighbour] == -1) {  // 邻居未访问
                levels[neighbour] = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}

/**
 * Kernel: Frontier Vertex-Centric BFS（基础版）
 * 
 * 使用稀疏前沿队列，只处理前沿中的顶点
 */
__global__ void bsf_frontier_vertex_centric_kernel(CSRGraph csrGraph, int* levels, int* prevFrontier, 
                                                   int* currFrontier, int numPrevFrontier, 
                                                   int* numCurrentFrontier, unsigned int currLevel) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; edge++) {
            unsigned int neighbour = csrGraph.dst[edge];
            // 使用atomicCAS避免重复访问
            if (atomicCAS(&levels[neighbour], -1, currLevel) == -1) {
                unsigned int currFrontierIdx = atomicAdd(numCurrentFrontier, 1);
                currFrontier[currFrontierIdx] = neighbour;
            }
        }
    }
}

/**
 * Kernel: Frontier Vertex-Centric BFS（带私有化）
 * 
 * 使用共享内存私有化减少全局原子操作
 */
__global__ void bsf_frontier_vertex_centric_with_privatization_kernel(CSRGraph csrGraph, int* levels, 
                                                                      int* prevFrontier, int* currFrontier, 
                                                                      int numPrevFrontier, int* numCurrFrontier, 
                                                                      int currLevel) {
    // 共享内存：局部前沿队列
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();
    
    //执行BFS
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            // 使用-1作为未访问标记（之前版本用UINT_MAX会导致问题）
            if (atomicCAS(&levels[neighbor], -1, currLevel) == -1) {
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY) {
                    // 添加到共享内存队列
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    // 共享内存满，直接写入全局内存
                    numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();
    
    // 分配全局队列空间
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();
    
    // 提交共享内存队列到全局内存
    for (unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s;
         currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

// ====================== Host端实现 ======================

/**
 * Push Vertex-Centric BFS（GPU实现）
 */
int* bfsParallelPushVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode) {
    // 创建并初始化level数组
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // 分配newVertexVisited标志
    int* d_newVertexVisited;
    CHECK_CUDA(cudaMalloc(&d_newVertexVisited, sizeof(int)));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // BFS主循环
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        CHECK_CUDA(cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice));
        
        bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost));
        currLevel++;
    }
    
    // 拷贝结果回主机
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // 释放GPU内存
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_newVertexVisited));
    
    return hostLevels;
}

/**
 * Pull Vertex-Centric BFS（GPU实现）
 */
int* bfsParallelPullVertexCentricDevice(const CSCGraph& deviceGraph, int startingNode) {
    // 创建并初始化level数组
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // 分配newVertexVisited标志
    int* d_newVertexVisited;
    CHECK_CUDA(cudaMalloc(&d_newVertexVisited, sizeof(int)));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // BFS主循环
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        CHECK_CUDA(cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice));
        
        bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost));
        currLevel++;
    }
    
    // 拷贝结果回主机
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // 释放GPU内存
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_newVertexVisited));
    
    return hostLevels;
}

/**
 * Edge-Centric BFS（GPU实现）
 */
int* bfsParallelEdgeCentricDevice(const COOGraph& deviceGraph, int startingNode) {
    // 创建并初始化level数组
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // 分配newVertexVisited标志
    int* d_newVertexVisited;
    CHECK_CUDA(cudaMalloc(&d_newVertexVisited, sizeof(int)));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceGraph.numEdges + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // BFS主循环
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        CHECK_CUDA(cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice));
        
        bsf_edge_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_newVertexVisited, currLevel);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost));
        currLevel++;
    }
    
    // 拷贝结果回主机
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // 释放GPU内存
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_newVertexVisited));
    
    return hostLevels;
}

/**
 * Frontier Vertex-Centric BFS（基础版，GPU实现）
 */
int* bfsParallelFrontierVertexCentricDevice(const CSRGraph& deviceGraph, int startingNode) {
    // 创建并初始化level数组
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // 初始化前沿队列（主机端）
    int* hostPrevFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    hostPrevFrontier[0] = startingNode;
    int hostNumPrevFrontier = 1;
    
    int* hostCurrFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    int hostNumCurrFrontier = 0;
    
    // 分配GPU前沿队列
    int *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
    CHECK_CUDA(cudaMalloc(&d_prevFrontier, sizeof(int) * deviceGraph.numVertices));
    CHECK_CUDA(cudaMalloc(&d_currFrontier, sizeof(int) * deviceGraph.numVertices));
    CHECK_CUDA(cudaMalloc(&d_numCurrFrontier, sizeof(int)));
    
    int currLevel = 1;
    
    // 前沿队列非空时继续BFS
    while (hostNumPrevFrontier > 0) {
        // 拷贝前沿到GPU
        CHECK_CUDA(cudaMemcpy(d_prevFrontier, hostPrevFrontier, 
                             sizeof(int) * hostNumPrevFrontier, cudaMemcpyHostToDevice));
        
        // 重置当前前沿计数
        hostNumCurrFrontier = 0;
        CHECK_CUDA(cudaMemcpy(d_numCurrFrontier, &hostNumCurrFrontier, sizeof(int), cudaMemcpyHostToDevice));
        
        // 启动kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (hostNumPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;
        
        bsf_frontier_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_prevFrontier, d_currFrontier, 
            hostNumPrevFrontier, d_numCurrFrontier, currLevel);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 获取新前沿大小
        CHECK_CUDA(cudaMemcpy(&hostNumCurrFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost));
        
        // 交换前沿队列
        if (hostNumCurrFrontier > 0) {
            CHECK_CUDA(cudaMemcpy(hostCurrFrontier, d_currFrontier, 
                                 sizeof(int) * hostNumCurrFrontier, cudaMemcpyDeviceToHost));
            
            int* tempFrontier = hostPrevFrontier;
            hostPrevFrontier = hostCurrFrontier;
            hostCurrFrontier = tempFrontier;
            
            hostNumPrevFrontier = hostNumCurrFrontier;
        } else {
            hostNumPrevFrontier = 0;
        }
        
        currLevel++;
    }
    
    // 拷贝结果回主机
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // 释放内存
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_prevFrontier));
    CHECK_CUDA(cudaFree(d_currFrontier));
    CHECK_CUDA(cudaFree(d_numCurrFrontier));
    free(hostPrevFrontier);
    free(hostCurrFrontier);
    
    return hostLevels;
}

/**
 * Frontier Vertex-Centric BFS（带私有化优化，GPU实现）
 */
int* bfsParallelFrontierVertexCentricOptimizedDevice(const CSRGraph& deviceGraph, int startingNode) {
    // 创建并初始化level数组
    int* d_levels = allocateAndInitLevelsOnDevice(deviceGraph.numVertices, startingNode);
    
    // 初始化前沿队列
    int* hostPrevFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    hostPrevFrontier[0] = startingNode;
    int hostNumPrevFrontier = 1;
    
    int* hostCurrFrontier = (int*)malloc(sizeof(int) * deviceGraph.numVertices);
    int hostNumCurrFrontier = 0;
    
    // 分配GPU前沿队列
    int *d_prevFrontier, *d_currFrontier, *d_numCurrFrontier;
    CHECK_CUDA(cudaMalloc(&d_prevFrontier, sizeof(int) * deviceGraph.numVertices));
    CHECK_CUDA(cudaMalloc(&d_currFrontier, sizeof(int) * deviceGraph.numVertices));
    CHECK_CUDA(cudaMalloc(&d_numCurrFrontier, sizeof(int)));
    
    int currLevel = 1;
    
    // 前沿队列非空时继续BFS
    while (hostNumPrevFrontier > 0) {
        CHECK_CUDA(cudaMemcpy(d_prevFrontier, hostPrevFrontier, 
                             sizeof(int) * hostNumPrevFrontier, cudaMemcpyHostToDevice));
        
        hostNumCurrFrontier = 0;
        CHECK_CUDA(cudaMemcpy(d_numCurrFrontier, &hostNumCurrFrontier, sizeof(int), cudaMemcpyHostToDevice));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (hostNumPrevFrontier + threadsPerBlock - 1) / threadsPerBlock;
        
        bsf_frontier_vertex_centric_with_privatization_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceGraph, d_levels, d_prevFrontier, d_currFrontier, 
            hostNumPrevFrontier, d_numCurrFrontier, currLevel);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(&hostNumCurrFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (hostNumCurrFrontier > 0) {
            CHECK_CUDA(cudaMemcpy(hostCurrFrontier, d_currFrontier, 
                                 sizeof(int) * hostNumCurrFrontier, cudaMemcpyDeviceToHost));
            
            int* tempFrontier = hostPrevFrontier;
            hostPrevFrontier = hostCurrFrontier;
            hostCurrFrontier = tempFrontier;
            
            hostNumPrevFrontier = hostNumCurrFrontier;
        } else {
            hostNumPrevFrontier = 0;
        }
        
        currLevel++;
    }
    
    // 拷贝结果回主机
    int* hostLevels = copyLevelsToHost(d_levels, deviceGraph.numVertices);
    
    // 释放内存
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_prevFrontier));
    CHECK_CUDA(cudaFree(d_currFrontier));
    CHECK_CUDA(cudaFree(d_numCurrFrontier));
    free(hostPrevFrontier);
    free(hostCurrFrontier);
    
    return hostLevels;
}

/**
 * Direction-Optimized BFS（方向优化，GPU实现）
 * 
 * 根据访问顶点比例动态切换Push/Pull策略
 */
int* bfsDirectionOptimizedDevice(const CSRGraph& deviceCSRGraph, const CSCGraph& deviceCSCGraph, 
                                 int startingNode, float alpha) {
    // 创建并初始化level数组
    int* hostLevels = (int*)malloc(sizeof(int) * deviceCSRGraph.numVertices);
    for (int i = 0; i < deviceCSRGraph.numVertices; i++) {
        hostLevels[i] = -1;
    }
    hostLevels[startingNode] = 0;
    
    // 分配GPU内存
    int *d_levels, *d_newVertexVisited;
    CHECK_CUDA(cudaMalloc(&d_levels, sizeof(int) * deviceCSRGraph.numVertices));
    CHECK_CUDA(cudaMalloc(&d_newVertexVisited, sizeof(int)));
    
    // 拷贝初始level到GPU
    CHECK_CUDA(cudaMemcpy(d_levels, hostLevels, sizeof(int) * deviceCSRGraph.numVertices, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (deviceCSRGraph.numVertices + threadsPerBlock - 1) / threadsPerBlock;
    
    int currLevel = 1;
    int hostNewVertexVisited = 1;
    
    // 追踪访问顶点数
    int totalVertices = deviceCSRGraph.numVertices;
    int visitedVertices = 1;  // 起点已访问
    bool usingPush = true;    // 初始使用Push策略
    
    // BFS主循环
    while (hostNewVertexVisited != 0) {
        hostNewVertexVisited = 0;
        CHECK_CUDA(cudaMemcpy(d_newVertexVisited, &hostNewVertexVisited, sizeof(int), cudaMemcpyHostToDevice));
        
        // 根据访问顶点比例决定策略
        float visitedFraction = (float)visitedVertices / totalVertices;
        if (usingPush && visitedFraction > alpha) {
            // 切换到Pull策略
            usingPush = false;
        }
        
        if (usingPush) {
            // 使用Push策略
            bsf_push_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                deviceCSRGraph, d_levels, d_newVertexVisited, currLevel);
        } else {
            // 使用Pull策略
            bsf_pull_vertex_centric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
                deviceCSCGraph, d_levels, d_newVertexVisited, currLevel);
        }
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaMemcpy(&hostNewVertexVisited, d_newVertexVisited, sizeof(int), cudaMemcpyDeviceToHost));
        
        // 更新访问顶点数
        if (hostNewVertexVisited) {
            CHECK_CUDA(cudaMemcpy(hostLevels, d_levels, sizeof(int) * totalVertices, cudaMemcpyDeviceToHost));
            int newVisitedCount = 0;
            for (int i = 0; i < totalVertices; i++) {
                if (hostLevels[i] != -1) {
                    newVisitedCount++;
                }
            }
            visitedVertices = newVisitedCount;
        }
        
        currLevel++;
    }
    
    // 拷贝最终结果
    CHECK_CUDA(cudaMemcpy(hostLevels, d_levels, sizeof(int) * totalVertices, cudaMemcpyDeviceToHost));
    
    // 释放GPU内存
    CHECK_CUDA(cudaFree(d_levels));
    CHECK_CUDA(cudaFree(d_newVertexVisited));
    
    return hostLevels;
}
