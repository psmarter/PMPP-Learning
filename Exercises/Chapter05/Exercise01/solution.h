#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 朴素矩阵乘法（无优化）
 * 每个线程直接从全局内存读取数据
 * 
 * @param h_P 输出矩阵（主机内存）
 * @param h_M 输入矩阵 M（主机内存）
 * @param h_N 输入矩阵 N（主机内存）
 * @param m 矩阵 M 的行数
 * @param n 矩阵 M 的列数 / 矩阵 N 的行数
 * @param o 矩阵 N 的列数
 */
void matrixMul(float* h_P, const float* h_M, const float* h_N, int m, int n, int o);

/**
 * Tiled 矩阵乘法（共享内存优化）
 * 使用共享内存分块加载数据，减少全局内存访问
 * 
 * @param h_P 输出矩阵（主机内存）
 * @param h_M 输入矩阵 M（主机内存）
 * @param h_N 输入矩阵 N（主机内存）
 * @param m 矩阵 M 的行数
 * @param n 矩阵 M 的列数 / 矩阵 N 的行数
 * @param o 矩阵 N 的列数
 */
void matrixMulTiled(float* h_P, const float* h_M, const float* h_N, int m, int n, int o);

#endif // SOLUTION_H
