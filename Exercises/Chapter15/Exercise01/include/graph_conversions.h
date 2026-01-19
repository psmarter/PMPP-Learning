#ifndef GRAPH_CONVERSIONS_H
#define GRAPH_CONVERSIONS_H

/**
 * 第十五章：图遍历 - 图格式转换
 * 
 * 提供COO、CSR、CSC之间的转换函数
 */

#include "graph_structures.h"

/**
 * COO 转 CSR
 * @param cooGraph COO格式的图
 * @return CSR格式的图
 */
CSRGraph convertCOOtoCSR(const COOGraph& cooGraph);

/**
 * CSR 转 COO
 * @param csrGraph CSR格式的图
 * @return COO格式的图
 */
COOGraph convertCSRtoCOO(const CSRGraph& csrGraph);

/**
 * COO 转 CSC
 * @param cooGraph COO格式的图
 * @return CSC格式的图
 */
CSCGraph convertCOOtoCSC(const COOGraph& cooGraph);

/**
 * CSR 转 CSC
 * 先转COO再转CSC
 * @param csrGraph CSR格式的图
 * @return CSC格式的图
 */
CSCGraph convertCSRtoCSC(const CSRGraph& csrGraph);

#endif // GRAPH_CONVERSIONS_H
