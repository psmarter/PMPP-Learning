#ifndef GRAPH_GENERATORS_H
#define GRAPH_GENERATORS_H

/**
 * 第十五章：图遍历 - 图生成器
 * 
 * 提供两种图生成算法：
 * 1. 无标度图 (Scale-Free Graph) - Barabási-Albert 模型
 * 2. 小世界图 (Small-World Graph) - Watts-Strogatz 模型
 */

#include "graph_structures.h"

/**
 * 生成无标度图（Barabási-Albert 模型）
 * 
 * 特点：少数顶点有大量连接，大多数顶点连接较少
 * 用于模拟社交网络、引用网络等
 * 
 * @param numVertices 顶点数
 * @param edgesPerNewVertex 每个新顶点添加的边数
 * @return COO格式的图
 */
COOGraph generateScaleFreeGraphCOO(int numVertices, int edgesPerNewVertex);

/**
 * 生成小世界图（Watts-Strogatz 模型）
 * 
 * 特点：高聚类系数、短平均路径长度
 * 用于模拟社会网络等
 * 
 * @param numVertices 顶点数
 * @param k 平均度数（必须是偶数）
 * @param rewireProbability 重连概率 (0-1)
 * @return COO格式的图
 */
COOGraph generateSmallWorldGraphCOO(int numVertices, int k, float rewireProbability);

#endif // GRAPH_GENERATORS_H
