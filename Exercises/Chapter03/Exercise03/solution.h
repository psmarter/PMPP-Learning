#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 标准矩阵乘法的设备函数声明（元素级 kernel）
 * 每个线程计算输出矩阵的一个元素
 * P = M × N
 * 
 * @param h_P 输出矩阵 P（主机内存）
 * @param h_M 输入矩阵 M（主机内存，m×n）
 * @param h_N 输入矩阵 N（主机内存，n×o）
 * @param m M 的行数
 * @param n M 的列数 / N 的行数
 * @param o N 的列数
 */
void matrixMultiplyDevice(float* h_P, const float* h_M, const float* h_N, 
                         int m, int n, int o);

#endif // SOLUTION_H
