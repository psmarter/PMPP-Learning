#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../../Common/utils.cuh"
#include "../../../Common/timer.h"
#include "solution.h"

/**
 * CPU 版本的矩阵乘法（用于验证正确性）
 * P = M × N
 */
void matrixMultiplyCPU(float* P, const float* M, const float* N, int size) {
    for (int row = 0; row < size; ++row) {
        for (int col = 0; col < size; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += M[row * size + k] * N[k * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}

/**
 * 验证两个矩阵是否相等（允许浮点误差）
 */
bool verifyResults(const float* gpu_result, const float* cpu_result, int size, float epsilon = 1e-3f) {
    int total_elements = size * size;
    for (int i = 0; i < total_elements; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) > epsilon) {
            printf("❌ Mismatch at element %d: GPU = %.6f, CPU = %.6f, diff = %.6f\n", 
                   i, gpu_result[i], cpu_result[i], fabs(gpu_result[i] - cpu_result[i]));
            return false;
        }
    }
    return true;
}

/**
 * 初始化矩阵为随机值
 */
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = (float)(rand() % 10);  // 0-9 之间的随机整数
    }
}

/**
 * 打印矩阵（仅用于小矩阵调试）
 */
void printMatrix(const float* matrix, int size, const char* name) {
    if (size > 10) {
        printf("%s (too large to print, size = %d×%d)\n", name, size, size);
        return;
    }
    printf("%s (%d×%d):\n", name, size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * 正确性测试
 */
bool testCorrectness(int size) {
    printf("\n=== 正确性测试 ===\n");
    printf("测试矩阵大小: %d×%d\n\n", size, size);
    
    // 分配主机内存
    size_t bytes = size * size * sizeof(float);
    float* h_M = (float*)malloc(bytes);
    float* h_N = (float*)malloc(bytes);
    float* h_P_row = (float*)malloc(bytes);
    float* h_P_col = (float*)malloc(bytes);
    float* h_P_cpu = (float*)malloc(bytes);
    
    // 初始化输入矩阵
    srand(2024);
    initializeMatrix(h_M, size);
    initializeMatrix(h_N, size);
    
    // 打印小矩阵（调试用）
    if (size <= 4) {
        printMatrix(h_M, size, "Matrix M");
        printMatrix(h_N, size, "Matrix N");
    }
    
    // 测试行级 kernel
    printf("测试行级 kernel...\n");
    matrixMultiplyRowDevice(h_P_row, h_M, h_N, size);
    
    // 测试列级 kernel
    printf("测试列级 kernel...\n");
    matrixMultiplyColDevice(h_P_col, h_M, h_N, size);
    
    // CPU 验证
    printf("计算 CPU 参考结果...\n");
    matrixMultiplyCPU(h_P_cpu, h_M, h_N, size);
    
    // 打印结果（小矩阵）
    if (size <= 4) {
        printMatrix(h_P_row, size, "Result (Row Kernel)");
        printMatrix(h_P_col, size, "Result (Col Kernel)");
        printMatrix(h_P_cpu, size, "Result (CPU)");
    }
    
    // 验证结果
    bool row_correct = verifyResults(h_P_row, h_P_cpu, size);
    bool col_correct = verifyResults(h_P_col, h_P_cpu, size);
    
    printf("\n行级 kernel: %s\n", row_correct ? "✅ 通过" : "❌ 失败");
    printf("列级 kernel: %s\n", col_correct ? "✅ 通过" : "❌ 失败");
    
    // 释放内存
    free(h_M);
    free(h_N);
    free(h_P_row);
    free(h_P_col);
    free(h_P_cpu);
    
    return row_correct && col_correct;
}

/**
 * 性能测试
 */
void testPerformance(int size, int iterations) {
    printf("\n=== 性能测试 ===\n");
    printf("矩阵大小: %d×%d\n", size, size);
    printf("迭代次数: %d\n\n", iterations);
    
    // 分配内存
    size_t bytes = size * size * sizeof(float);
    float* h_M = (float*)malloc(bytes);
    float* h_N = (float*)malloc(bytes);
    float* h_P = (float*)malloc(bytes);
    
    initializeMatrix(h_M, size);
    initializeMatrix(h_N, size);
    
    // 预热
    matrixMultiplyRowDevice(h_P, h_M, h_N, size);
    
    // 测试行级 kernel
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        matrixMultiplyRowDevice(h_P, h_M, h_N, size);
    }
    timer.stop();
    
    double row_time = timer.elapsed_ms() / iterations;
    double operations = 2.0 * size * size * size;  // 矩阵乘法的浮点操作数
    double row_gflops = (operations / row_time) / 1e6;  // GFLOPS
    
    printf("行级 kernel:\n");
    printf("  平均时间: %.3f ms\n", row_time);
    printf("  性能: %.2f GFLOPS\n\n", row_gflops);
    
    // 测试列级 kernel
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        matrixMultiplyColDevice(h_P, h_M, h_N, size);
    }
    timer.stop();
    
    double col_time = timer.elapsed_ms() / iterations;
    double col_gflops = (operations / col_time) / 1e6;
    
    printf("列级 kernel:\n");
    printf("  平均时间: %.3f ms\n", col_time);
    printf("  性能: %.2f GFLOPS\n\n", col_gflops);
    
    // 释放内存
    free(h_M);
    free(h_N);
    free(h_P);
}

int main() {
    printf("================================================\n");
    printf("第三章 - 练习1: 矩阵乘法（行级和列级 kernel）\n");
    printf("================================================\n");
    
    // 打印设备信息
    printDeviceInfo();
    
    // 正确性测试（小矩阵）
    if (!testCorrectness(64)) {
        printf("\n❌ 正确性测试失败！\n");
        return 1;
    }
    
    // 性能测试（大矩阵）
    testPerformance(512, 10);
    
    printf("✅ 所有测试完成！\n\n");
    
    return 0;
}
