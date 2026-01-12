#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 向量乘法的设备函数声明
 * 在 solution.cu 中实现，在 test.cpp 中调用
 * 
 * @param h_c 输出向量（主机内存）
 * @param h_a 输入向量 A（主机内存）
 * @param h_b 输入向量 B（主机内存）
 * @param n   向量长度
 */
void vectorMultiplyDevice(float* h_c, const float* h_a, const float* h_b, int n);

#endif // SOLUTION_H
