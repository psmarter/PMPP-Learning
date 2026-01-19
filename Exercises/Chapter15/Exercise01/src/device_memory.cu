/**
 * 第十五章：图遍历 - GPU内存管理实现
 * 
 * 参考：chapter-15/code/src/device_memory.cu
 * 
 * 提供图数据结构在GPU上的分配、释放和数据传输功能
 */

#include "../include/device_memory.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdlib>

// ====================== CSR图内存管理 ======================

/**
 * 在GPU上分配CSR图
 * 
 * @param hostGraph 主机端CSR图
 * @return GPU上的CSR图结构
 */
CSRGraph allocateCSRGraphOnDevice(const CSRGraph& hostGraph) {
    int *d_srcPtrs, *d_dst, *d_values;
    
    // 计算数组大小
    size_t srcPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t dstSize = sizeof(int) * hostGraph.srcPtrs[hostGraph.numVertices];  // 总边数
    
    // 分配GPU内存
    CHECK_CUDA(cudaMalloc(&d_srcPtrs, srcPtrsSize));
    CHECK_CUDA(cudaMalloc(&d_dst, dstSize));
    CHECK_CUDA(cudaMalloc(&d_values, dstSize));
    
    // 拷贝数据到GPU
    CHECK_CUDA(cudaMemcpy(d_srcPtrs, hostGraph.srcPtrs, srcPtrsSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dst, hostGraph.dst, dstSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, hostGraph.values, dstSize, cudaMemcpyHostToDevice));
    
    // 返回GPU上的CSR图结构
    CSRGraph deviceGraph;
    deviceGraph.srcPtrs = d_srcPtrs;
    deviceGraph.dst = d_dst;
    deviceGraph.values = d_values;
    deviceGraph.numVertices = hostGraph.numVertices;
    
    return deviceGraph;
}

/**
 * 释放GPU上的CSR图
 * 
 * @param deviceGraph GPU上的CSR图指针
 */
void freeCSRGraphOnDevice(CSRGraph* deviceGraph) {
    if (deviceGraph->srcPtrs) {
        CHECK_CUDA(cudaFree(deviceGraph->srcPtrs));
        deviceGraph->srcPtrs = nullptr;
    }
    if (deviceGraph->dst) {
        CHECK_CUDA(cudaFree(deviceGraph->dst));
        deviceGraph->dst = nullptr;
    }
    if (deviceGraph->values) {
        CHECK_CUDA(cudaFree(deviceGraph->values));
        deviceGraph->values = nullptr;
    }
}

// ====================== CSC图内存管理 ======================

/**
 * 在GPU上分配CSC图
 * 
 * @param hostGraph 主机端CSC图
 * @return GPU上的CSC图结构
 */
CSCGraph allocateCSCGraphOnDevice(const CSCGraph& hostGraph) {
    int *d_dstPtrs, *d_src, *d_values;
    
    // 计算数组大小
    size_t dstPtrsSize = sizeof(int) * (hostGraph.numVertices + 1);
    size_t srcSize = sizeof(int) * hostGraph.dstPtrs[hostGraph.numVertices];  // 总边数
    
    // 分配GPU内存
    CHECK_CUDA(cudaMalloc(&d_dstPtrs, dstPtrsSize));
    CHECK_CUDA(cudaMalloc(&d_src, srcSize));
    CHECK_CUDA(cudaMalloc(&d_values, srcSize));
    
    // 拷贝数据到GPU
    CHECK_CUDA(cudaMemcpy(d_dstPtrs, hostGraph.dstPtrs, dstPtrsSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_src, hostGraph.src, srcSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, hostGraph.values, srcSize, cudaMemcpyHostToDevice));
    
    // 返回GPU上的CSC图结构
    CSCGraph deviceGraph;
    deviceGraph.dstPtrs = d_dstPtrs;
    deviceGraph.src = d_src;
    deviceGraph.values = d_values;
    deviceGraph.numVertices = hostGraph.numVertices;
    
    return deviceGraph;
}

/**
 * 释放GPU上的CSC图
 * 
 * @param deviceGraph GPU上的CSC图指针
 */
void freeCSCGraphOnDevice(CSCGraph* deviceGraph) {
    if (deviceGraph->dstPtrs) {
        CHECK_CUDA(cudaFree(deviceGraph->dstPtrs));
        deviceGraph->dstPtrs = nullptr;
    }
    if (deviceGraph->src) {
        CHECK_CUDA(cudaFree(deviceGraph->src));
        deviceGraph->src = nullptr;
    }
    if (deviceGraph->values) {
        CHECK_CUDA(cudaFree(deviceGraph->values));
        deviceGraph->values = nullptr;
    }
}

// ====================== COO图内存管理 ======================

/**
 * 在GPU上分配COO图
 * 
 * @param hostGraph 主机端COO图
 * @return GPU上的COO图结构
 */
COOGraph allocateCOOGraphOnDevice(const COOGraph& hostGraph) {
    int *d_scr, *d_dst, *d_values;
    
    // 计算数组大小
    size_t edgeArraysSize = sizeof(int) * hostGraph.numEdges;
    
    // 分配GPU内存
    CHECK_CUDA(cudaMalloc(&d_scr, edgeArraysSize));
    CHECK_CUDA(cudaMalloc(&d_dst, edgeArraysSize));
    CHECK_CUDA(cudaMalloc(&d_values, edgeArraysSize));
    
    // 拷贝数据到GPU
    CHECK_CUDA(cudaMemcpy(d_scr, hostGraph.scr, edgeArraysSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dst, hostGraph.dst, edgeArraysSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, hostGraph.values, edgeArraysSize, cudaMemcpyHostToDevice));
    
    // 返回GPU上的COO图结构
    COOGraph deviceGraph;
    deviceGraph.scr = d_scr;
    deviceGraph.dst = d_dst;
    deviceGraph.values = d_values;
    deviceGraph.numEdges = hostGraph.numEdges;
    deviceGraph.numVertices = hostGraph.numVertices;
    
    return deviceGraph;
}

/**
 * 释放GPU上的COO图
 * 
 * @param deviceGraph GPU上的COO图指针
 */
void freeCOOGraphOnDevice(COOGraph* deviceGraph) {
    if (deviceGraph->scr) {
        CHECK_CUDA(cudaFree(deviceGraph->scr));
        deviceGraph->scr = nullptr;
    }
    if (deviceGraph->dst) {
        CHECK_CUDA(cudaFree(deviceGraph->dst));
        deviceGraph->dst = nullptr;
    }
    if (deviceGraph->values) {
        CHECK_CUDA(cudaFree(deviceGraph->values));
        deviceGraph->values = nullptr;
    }
}

// ====================== Level数组管理 ======================

/**
 * 在GPU上分配并初始化level数组
 * 
 * @param numVertices 顶点数
 * @param startVertex 起始顶点
 * @return GPU上的level数组
 */
int* allocateAndInitLevelsOnDevice(int numVertices, int startVertex) {
    // 在主机端创建并初始化level数组
    int* hostLevels = (int*)malloc(sizeof(int) * numVertices);
    for (int i = 0; i < numVertices; i++) {
        hostLevels[i] = -1;  // 未访问标记为-1
    }
    hostLevels[startVertex] = 0;  // 起点level为0
    
    // 在GPU上分配内存
    int* deviceLevels;
    CHECK_CUDA(cudaMalloc(&deviceLevels, sizeof(int) * numVertices));
    
    // 拷贝初始化数据到GPU
    CHECK_CUDA(cudaMemcpy(deviceLevels, hostLevels, sizeof(int) * numVertices, cudaMemcpyHostToDevice));
    
    // 释放主机端临时数组
    free(hostLevels);
    
    return deviceLevels;
}

/**
 * 将level数组从GPU拷贝回主机
 * 
 * @param d_levels GPU上的level数组
 * @param numVertices 顶点数
 * @return 主机端的level数组（需要调用者释放）
 */
int* copyLevelsToHost(int* d_levels, int numVertices) {
    int* hostLevels = (int*)malloc(sizeof(int) * numVertices);
    CHECK_CUDA(cudaMemcpy(hostLevels, d_levels, sizeof(int) * numVertices, cudaMemcpyDeviceToHost));
    return hostLevels;
}
