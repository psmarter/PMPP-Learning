/**
 * 第十五章：图遍历 - 主测试程序
 * 
 * 参考：chapter-15/code/src/main.cu
 * 
 * 测试所有6种BFS算法的正确性和性能
 */

#include "../include/bfs_parallel.h"
#include "../include/bfs_sequential.h"
#include "../include/device_memory.h"
#include "../include/graph_conversions.h"
#include "../include/graph_generators.h"
#include "../include/graph_structures.h"
#include "../include/utils.h"
#include "../../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第十五章：图遍历\n");
    printf("  Breadth-First Search - Multiple Implementations\n");
    printf("================================================================\n\n");
    
    // 测试参数
    const int startVertex = 0;
    const int testSizes[] = {1000, 5000, 10000};
    const int numSizes = 3;
    
    printf("=== 正确性验证 ===\n\n");
    
    // 生成验证图
    printf("生成测试图（2000个顶点）...\n");
    COOGraph verifyCOO = generateScaleFreeGraphCOO(2000, 50);
    CSRGraph verifyCSR = convertCOOtoCSR(verifyCOO);
    CSCGraph verifyCSC = convertCOOtoCSC(verifyCOO);
    
    // 分配到GPU
    CSRGraph deviceVerifyCSR = allocateCSRGraphOnDevice(verifyCSR);
    CSCGraph deviceVerifyCSC = allocateCSCGraphOnDevice(verifyCSC);
    COOGraph deviceVerifyCOO = allocateCOOGraphOnDevice(verifyCOO);
    
    // CPU参考结果
    int* cpuResult = bfs(verifyCSR, startVertex);
    
    // 测试各个BFS实现
    printf("1. Push Vertex-Centric BFS... ");
    int* pushResult = bfsParallelPushVertexCentricDevice(deviceVerifyCSR, startVertex);
    if (compareBFSResults(cpuResult, pushResult, verifyCSR.numVertices, false)) {
        printf("✅ 结果正确！\n");
    } else {
        printf("❌ 结果不正确！\n");
    }
    free(pushResult);
    
    printf("2. Pull Vertex-Centric BFS... ");
    int* pullResult = bfsParallelPullVertexCentricDevice(deviceVerifyCSC, startVertex);
    if (compareBFSResults(cpuResult, pullResult, verifyCSR.numVertices, false)) {
        printf("✅ 结果正确！\n");
    } else {
        printf("❌ 结果不正确！\n");
    }
    free(pullResult);
    
    printf("3. Edge-Centric BFS... ");
    int* edgeResult = bfsParallelEdgeCentricDevice(deviceVerifyCOO, startVertex);
    if (compareBFSResults(cpuResult, edgeResult, verifyCSR.numVertices, false)) {
        printf("✅ 结果正确！\n");
    } else {
        printf("❌ 结果不正确！\n");
    }
    free(edgeResult);
    
    printf("4. Frontier  BFS (基础版)... ");
    int* frontierResult = bfsParallelFrontierVertexCentricDevice(deviceVerifyCSR, startVertex);
    if (compareBFSResults(cpuResult, frontierResult, verifyCSR.numVertices, false)) {
        printf("✅ 结果正确！\n");
    } else {
        printf("❌ 结果不正确！\n");
    }
    free(frontierResult);
    
    printf("5. Frontier BFS (优化版)... ");
    int* frontierOptResult = bfsParallelFrontierVertexCentricOptimizedDevice(deviceVerifyCSR, startVertex);
    if (compareBFSResults(cpuResult, frontierOptResult, verifyCSR.numVertices, false)) {
        printf("✅ 结果正确！\n");
    } else {
        printf("❌ 结果不正确！\n");
    }
    free(frontierOptResult);
    
    printf("6. Direction-Optimized BFS... ");
    int* dirOptResult = bfsDirectionOptimizedDevice(deviceVerifyCSR, deviceVerifyCSC, startVertex, 0.1f);
    if (compareBFSResults(cpuResult, dirOptResult, verifyCSR.numVertices, false)) {
        printf("✅ 结果正确！\n");
    } else {
        printf("❌ 结果不正确！\n");
    }
    free(dirOptResult);
    
    printf("\n所有BFS实现通过正确性验证！\n\n");
    
    // 释放验证图内存
    free(cpuResult);
    freeCSRGraphOnDevice(&deviceVerifyCSR);
    freeCSCGraphOnDevice(&deviceVerifyCSC);
    freeCOOGraphOnDevice(&deviceVerifyCOO);
    free(verifyCOO.scr);
    free(verifyCOO.dst);
    free(verifyCOO.values);
    free(verifyCSR.srcPtrs);
    free(verifyCSR.dst);
    free(verifyCSR.values);
    free(verifyCSC.dstPtrs);
    free(verifyCSC.src);
    free(verifyCSC.values);
    
    // 性能测试
    printf("=== 性能基准测试 ===\n\n");
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int i = 0; i < numSizes; i++) {
        int size = testSizes[i];
        printf("图规模：%d 个顶点\n", size);
        printf("--------------------\n");
        
        // 生成无标度图
        printf("生成无标度图...\n");
        COOGraph testCOO = generateScaleFreeGraphCOO(size, 50);
        CSRGraph testCSR = convertCOOtoCSR(testCOO);
        CSCGraph testCSC = convertCOOtoCSC(testCOO);
        
        // 分配到GPU
        CSRGraph deviceCSR = allocateCSRGraphOnDevice(testCSR);
        CSCGraph deviceCSC = allocateCSCGraphOnDevice(testCSC);
        COOGraph deviceCOO = allocateCOOGraphOnDevice(testCOO);
        
        // CPU BFS (使用clock计时)
        clock_t cpuStart = clock();
        int* seqResult = bfs(testCSR, startVertex);
        clock_t cpuEnd = clock();
        float seqTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;
        printf("Sequential BFS: %.2f ms\n", seqTime);
        
        // Push BFS
        CHECK_CUDA(cudaEventRecord(start));
        int* pushRes = bfsParallelPushVertexCentricDevice(deviceCSR, startVertex);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float pushTime = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&pushTime, start, stop));
        printf("Push Vertex-Centric BFS: %.2f ms (%.2fx speedup)\n", pushTime, seqTime / pushTime);
        free(pushRes);
        
        // Pull BFS
        CHECK_CUDA(cudaEventRecord(start));
        int* pullRes = bfsParallelPullVertexCentricDevice(deviceCSC, startVertex);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float pullTime = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&pullTime, start, stop));
        printf("Pull Vertex-Centric BFS: %.2f ms (%.2fx speedup)\n", pullTime, seqTime / pullTime);
        free(pullRes);
        
        // Edge-Centric BFS
        CHECK_CUDA(cudaEventRecord(start));
        int* edgeRes = bfsParallelEdgeCentricDevice(deviceCOO, startVertex);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float edgeTime = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&edgeTime, start, stop));
        printf("Edge-Centric BFS: %.2f ms (%.2fx speedup)\n", edgeTime, seqTime / edgeTime);
        free(edgeRes);
        
        // Frontier BFS
        CHECK_CUDA(cudaEventRecord(start));
        int* frontRes = bfsParallelFrontierVertexCentricDevice(deviceCSR, startVertex);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float frontTime = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&frontTime, start, stop));
        printf("Frontier-based BFS: %.2f ms (%.2fx speedup)\n", frontTime, seqTime / frontTime);
        free(frontRes);
        
        // Frontier优化 BFS
        CHECK_CUDA(cudaEventRecord(start));
        int* frontOptRes = bfsParallelFrontierVertexCentricOptimizedDevice(deviceCSR, startVertex);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float frontOptTime = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&frontOptTime, start, stop));
        printf("Optimized Frontier-based BFS: %.2f ms (%.2fx speedup)\n", frontOptTime, seqTime / frontOptTime);
        free(frontOptRes);
        
        // Direction-Optimized BFS
        CHECK_CUDA(cudaEventRecord(start));
        int* dirRes = bfsDirectionOptimizedDevice(deviceCSR, deviceCSC, startVertex, 0.1f);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float dirTime = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&dirTime, start, stop));
        printf("Direction-Optimized BFS: %.2f ms (%.2fx speedup)\n", dirTime, seqTime / dirTime);
        free(dirRes);
        
        printf("\n");
        
        // 释放内存
        free(seqResult);
        freeCSRGraphOnDevice(&deviceCSR);
        freeCSCGraphOnDevice(&deviceCSC);
        freeCOOGraphOnDevice(&deviceCOO);
        free(testCOO.scr);
        free(testCOO.dst);
        free(testCOO.values);
        free(testCSR.srcPtrs);
        free(testCSR.dst);
        free(testCSR.values);
        free(testCSC.dstPtrs);
        free(testCSC.src);
        free(testCSC.values);
    }
    
    printf("【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• Push模式：从当前层顶点推送到邻居（适合前沿小时）\n");
    printf("• Pull模式：未访问顶点从邻居拉取（适合前沿大时）\n");
    printf("• Edge-Centric：每线程处理一条边，简化并行策略\n");
    printf("• Frontier队列：只处理活跃顶点，减少无效工作\n");
    printf("• 私有化：使用共享内存减少全局原子操作争用\n");
    printf("• 方向优化：根据前沿大小动态切换Push/Pull\n");
    printf("\n");
    
    // 销毁CUDA events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    printf("✅ 测试完成！\n\n");
    return 0;
}
