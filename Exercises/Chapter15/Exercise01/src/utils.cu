/**
 * 第十五章：图遍历 - 工具函数实现
 * 
 * 参考：chapter-15/code/src/utils.cu
 */

#include "../include/utils.h"
#include <cstdio>
#include <cstdlib>

/**
 * 比较两个BFS结果是否相同
 * 
 * @param result1 第一个BFS结果（level数组）
 * @param result2 第二个BFS结果（level数组）
 * @param numVertices 顶点数
 * @param verbose 是否输出详细错误信息
 * @return true 如果所有顶点的level相同
 */
bool compareBFSResults(const int* result1, const int* result2, int numVertices, bool verbose) {
    bool allMatch = true;
    int mismatchCount = 0;
    
    for (int i = 0; i < numVertices; i++) {
        if (result1[i] != result2[i]) {
            allMatch = false;
            mismatchCount++;
            
            if (verbose && mismatchCount <= 10) {
                printf("  不匹配：顶点 %d: %d vs %d\n", i, result1[i], result2[i]);
            }
        }
    }
    
    if (!allMatch && verbose) {
        printf("  总计 %d 个顶点不匹配 (共 %d 个顶点)\n", mismatchCount, numVertices);
    }
    
    return allMatch;
}
