#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../../Common/utils.cuh"
#include "../../../Common/timer.h"
#include "solution.h"

/**
 * CPU 版本的矩阵向量乘法（用于验证正确性）
 * A = B × C
 */
void matrixVectorMultiplyCPU(float* A, const float* B, const float* C, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += B[i * cols + j] * C[j];
        }
        A[i] = sum;
    }
}

/**
 * 验证两个向量是否相等（允许浮点误差）
 */
bool verifyResults(const float* gpu_result, const float* cpu_result, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size; ++i) {
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
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (float)(rand() % 10);  // 0-9 之间的随机整数
    }
}

/**
 * 初始化向量为随机值
 */
void initializeVector(float* vector, int size) {
    for (int i = 0; i < size; ++i) {
        vector[i] = (float)(rand() % 10);
    }
}

/**
 * 打印矩阵（仅用于小矩阵调试）
 */
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    if (rows > 10 || cols > 10) {
        printf("%s (too large to print, size = %d×%d)\n", name, rows, cols);
        return;
    }
    printf("%s (%d×%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * 打印向量（仅用于小向量调试）
 */
void printVector(const float* vector, int size, const char* name) {
    if (size > 10) {
        printf("%s (too large to print, size = %d)\n", name, size);
        return;
    }
    printf("%s (%d): ", name, size);
    for (int i = 0; i < size; ++i) {
        printf("%.2f ", vector[i]);
    }
    printf("\n\n");
}

/**
 * 正确性测试
 */
bool testCorrectness(int rows, int cols) {
    printf("\n=== 正确性测试 ===\n");
    printf("矩阵大小: %d×%d, 向量大小: %d\n\n", rows, cols, cols);
    
    // 分配主机内存
    float* h_B = (float*)malloc(rows * cols * sizeof(float));
    float* h_C = (float*)malloc(cols * sizeof(float));
    float* h_A_gpu = (float*)malloc(rows * sizeof(float));
    float* h_A_cpu = (float*)malloc(rows * sizeof(float));
    
    // 初始化输入
    srand(2024);
    initializeMatrix(h_B, rows, cols);
    initializeVector(h_C, cols);
    
    // 打印小数据（调试用）
    if (rows <= 5 && cols <= 5) {
        printMatrix(h_B, rows, cols, "Matrix B");
        printVector(h_C, cols, "Vector C");
    }
    
    // GPU 计算
    printf("执行 GPU 计算...\n");
    matrixVectorMultiplyDevice(h_A_gpu, h_B, h_C, rows, cols);
    
    // CPU 验证
    printf("执行 CPU 验证...\n");
    matrixVectorMultiplyCPU(h_A_cpu, h_B, h_C, rows, cols);
    
    // 打印结果（小数据）
    if (rows <= 10) {
        printVector(h_A_gpu, rows, "Result (GPU)");
        printVector(h_A_cpu, rows, "Result (CPU)");
    }
    
    // 验证结果
    bool correct = verifyResults(h_A_gpu, h_A_cpu, rows);
    
    printf("%s\n", correct ? "✅ 正确性测试通过" : "❌ 正确性测试失败");
    
    // 释放内存
    free(h_B);
    free(h_C);
    free(h_A_gpu);
    free(h_A_cpu);
    
    return correct;
}

/**
 * 性能测试
 */
void testPerformance(int rows, int cols, int iterations) {
    printf("\n=== 性能测试 ===\n");
    printf("矩阵大小: %d×%d, 向量大小: %d\n", rows, cols, cols);
    printf("迭代次数: %d\n\n", iterations);
    
    // 分配内存
    float* h_B = (float*)malloc(rows * cols * sizeof(float));
    float* h_C = (float*)malloc(cols * sizeof(float));
    float* h_A = (float*)malloc(rows * sizeof(float));
    
    initializeMatrix(h_B, rows, cols);
    initializeVector(h_C, cols);
    
    // 预热
    matrixVectorMultiplyDevice(h_A, h_B, h_C, rows, cols);
    
    // 性能测试
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        matrixVectorMultiplyDevice(h_A, h_B, h_C, rows, cols);
    }
    timer.stop();
    
    double avg_time = timer.elapsed_ms() / iterations;
    double operations = 2.0 * rows * cols;  // 每个元素需要一次乘法和一次加法
    double gflops = (operations / avg_time) / 1e6;  // GFLOPS
    
    size_t bytes_read = (rows * cols + cols) * sizeof(float);  // 读取矩阵和向量
    size_t bytes_written = rows * sizeof(float);  // 写入结果向量
    double bandwidth = ((bytes_read + bytes_written) / avg_time) / 1e6;  // GB/s
    
    printf("结果:\n");
    printf("  平均时间: %.3f ms\n", avg_time);
    printf("  性能: %.2f GFLOPS\n", gflops);
    printf("  带宽: %.2f GB/s\n", bandwidth);
    
    // 释放内存
    free(h_B);
    free(h_C);
    free(h_A);
}

int main() {
    printf("==================================================\n");
    printf("第三章 - 练习2: 矩阵向量乘法\n");
    printf("==================================================\n");
    
    // 打印设备信息
    printDeviceInfo();
    
    // 正确性测试（小规模）
    if (!testCorrectness(128, 128)) {
        printf("\n❌ 正确性测试失败！\n");
        return 1;
    }
    
    // 性能测试（大规模）
    testPerformance(4096, 4096, 100);
    
    printf("\n✅ 所有测试完成！\n\n");
    
    return 0;
}
