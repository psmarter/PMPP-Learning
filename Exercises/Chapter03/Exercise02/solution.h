#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 矩阵向量乘法的设备函数声明
 * A = B × C （A是向量，B是矩阵，C是向量）
 * 每个线程计算输出向量的一个元素
 * 
 * @param h_A 输出向量 A（主机内存）
 * @param h_B 输入矩阵 B（主机内存）
 * @param h_C 输入向量 C（主机内存）
 * @param matrix_rows 矩阵行数
 * @param matrix_cols 矩阵列数（= 向量长度）
 */
void matrixVectorMultiplyDevice(float* h_A, const float* h_B, const float* h_C, 
                               int matrix_rows, int matrix_cols);

#endif // SOLUTION_H
