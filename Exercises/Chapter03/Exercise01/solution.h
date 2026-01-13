#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 矩阵乘法的设备函数声明（行级 kernel）
 * 每个线程计算输出矩阵的一整行
 * 
 * @param h_P 输出矩阵 P（主机内存）
 * @param h_M 输入矩阵 M（主机内存）
 * @param h_N 输入矩阵 N（主机内存）
 * @param size 矩阵大小（假设为方阵 size × size）
 */
void matrixMultiplyRowDevice(float* h_P, const float* h_M, const float* h_N, int size);

/**
 * 矩阵乘法的设备函数声明（列级 kernel）
 * 每个线程计算输出矩阵的一整列
 * 
 * @param h_P 输出矩阵 P（主机内存）
 * @param h_M 输入矩阵 M（主机内存）
 * @param h_N 输入矩阵 N（主机内存）
 * @param size 矩阵大小（假设为方阵 size × size）
 */
void matrixMultiplyColDevice(float* h_P, const float* h_M, const float* h_N, int size);

#endif // SOLUTION_H
