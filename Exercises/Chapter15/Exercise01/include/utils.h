#ifndef UTILS_H
#define UTILS_H

/**
 * 第十五章：图遍历 - 工具函数
 * 
 * 提供BFS结果比较等辅助功能
 */

/**
 * 比较两个BFS结果是否相同
 * @param result1 第一个BFS结果
 * @param result2 第二个BFS结果
 * @param numVertices 顶点数
 * @param verbose 是否输出详细信息
 * @return true 如果结果相同
 */
bool compareBFSResults(const int* result1, const int* result2, int numVertices, bool verbose = false);

#endif // UTILS_H
