/**
 * 第十五章：图遍历 - 图格式转换实现
 * 
 * 参考：chapter-15/code/src/graph_conversions.cu
 * 
 * 实现 COO、CSR、CSC 之间的转换
 */

#include "../include/graph_conversions.h"
#include "../include/graph_structures.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

/**
 * COO 转 CSR
 * 
 * 算法步骤：
 * 1. 统计每个源顶点的出边数
 * 2. 计算前缀和得到行指针
 * 3. 按顺序填充目标顶点和权值
 * 
 * @param cooGraph COO 格式的图
 * @return CSR 格式的图
 */
CSRGraph convertCOOtoCSR(const COOGraph& cooGraph) {
    CSRGraph csrGraph;
    csrGraph.numVertices = cooGraph.numVertices;
    csrGraph.srcPtrs = (int*)malloc(sizeof(int) * (cooGraph.numVertices + 1));
    csrGraph.dst = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    csrGraph.values = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    
    // 步骤1：初始化行指针为0
    for (int i = 0; i <= cooGraph.numVertices; i++) {
        csrGraph.srcPtrs[i] = 0;
    }
    
    // 步骤2：统计每个源顶点的出边数
    for (int i = 0; i < cooGraph.numEdges; i++) {
        csrGraph.srcPtrs[cooGraph.scr[i] + 1]++;
    }
    
    // 步骤3：前缀和得到行指针
    for (int i = 1; i <= cooGraph.numVertices; i++) {
        csrGraph.srcPtrs[i] += csrGraph.srcPtrs[i - 1];
    }
    
    // 步骤4：填充数据
    int* pos = (int*)malloc(sizeof(int) * cooGraph.numVertices);
    memcpy(pos, csrGraph.srcPtrs, sizeof(int) * cooGraph.numVertices);
    
    for (int i = 0; i < cooGraph.numEdges; i++) {
        int row = cooGraph.scr[i];
        int idx = pos[row]++;
        
        csrGraph.dst[idx] = cooGraph.dst[i];
        csrGraph.values[idx] = cooGraph.values[i];
    }
    
    free(pos);
    return csrGraph;
}

/**
 * CSR 转 COO
 * 
 * 直接展开 CSR 的压缩表示
 * 
 * @param csrGraph CSR 格式的图
 * @return COO 格式的图
 */
COOGraph convertCSRtoCOO(const CSRGraph& csrGraph) {
    int numVertices = csrGraph.numVertices;
    int numEdges = csrGraph.srcPtrs[numVertices];
    
    // 分配 COO 图内存
    COOGraph cooGraph;
    cooGraph.numVertices = numVertices;
    cooGraph.numEdges = numEdges;
    cooGraph.scr = (int*)malloc(sizeof(int) * numEdges);
    cooGraph.dst = (int*)malloc(sizeof(int) * numEdges);
    cooGraph.values = (int*)malloc(sizeof(int) * numEdges);
    
    // 展开 CSR
    int edgeIdx = 0;
    for (int i = 0; i < numVertices; i++) {
        for (int j = csrGraph.srcPtrs[i]; j < csrGraph.srcPtrs[i + 1]; j++) {
            cooGraph.scr[edgeIdx] = i;                      // 源顶点
            cooGraph.dst[edgeIdx] = csrGraph.dst[j];        // 目标顶点
            cooGraph.values[edgeIdx] = csrGraph.values[j];  // 边权值
            edgeIdx++;
        }
    }
    
    return cooGraph;
}

/**
 * COO 转 CSC
 * 
 * 算法步骤：
 * 1. 统计每个目标顶点的入边数
 * 2. 计算前缀和得到列指针
 * 3. 按顺序填充源顶点和权值
 * 
 * @param cooGraph COO 格式的图
 * @return CSC 格式的图
 */
CSCGraph convertCOOtoCSC(const COOGraph& cooGraph) {
    CSCGraph cscGraph;
    cscGraph.numVertices = cooGraph.numVertices;
    cscGraph.dstPtrs = (int*)malloc(sizeof(int) * (cooGraph.numVertices + 1));
    cscGraph.src = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    cscGraph.values = (int*)malloc(sizeof(int) * cooGraph.numEdges);
    
    // 步骤1：初始化列指针为0
    for (int i = 0; i <= cooGraph.numVertices; i++) {
        cscGraph.dstPtrs[i] = 0;
    }
    
    // 步骤2：统计每个目标顶点的入边数
    for (int i = 0; i < cooGraph.numEdges; i++) {
        cscGraph.dstPtrs[cooGraph.dst[i] + 1]++;
    }
    
    // 步骤3：前缀和得到列指针
    for (int i = 1; i <= cooGraph.numVertices; i++) {
        cscGraph.dstPtrs[i] += cscGraph.dstPtrs[i - 1];
    }
    
    // 步骤4：填充数据
    int* pos = (int*)malloc(sizeof(int) * cooGraph.numVertices);
    memcpy(pos, cscGraph.dstPtrs, sizeof(int) * cooGraph.numVertices);
    
    for (int i = 0; i < cooGraph.numEdges; i++) {
        int col = cooGraph.dst[i];
        int idx = pos[col]++;
        
        cscGraph.src[idx] = cooGraph.scr[i];
        cscGraph.values[idx] = cooGraph.values[i];
    }
    
    free(pos);
    return cscGraph;
}

/**
 * CSR 转 CSC
 * 
 * 通过 COO 中转
 * 
 * @param csrGraph CSR 格式的图
 * @return CSC 格式的图
 */
CSCGraph convertCSRtoCSC(const CSRGraph& csrGraph) {
    COOGraph cooGraph = convertCSRtoCOO(csrGraph);
    CSCGraph cscGraph = convertCOOtoCSC(cooGraph);
    
    // 释放临时 COO 图
    free(cooGraph.scr);
    free(cooGraph.dst);
    free(cooGraph.values);
    
    return cscGraph;
}
