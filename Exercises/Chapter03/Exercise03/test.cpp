#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../../Common/utils.cuh"
#include "../../../Common/timer.h"
#include "solution.h"

/**
 * CPU 版本的矩阵乘法（用于验证正确性）
 */
void matrixMultiplyCPU(float* P, const float* M, const float* N, int m, int n, int o) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < o; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += M[row * n + k] * N[k * o + col];
            }
            P[row * o + col] = sum;
        }
    }
}

/**
 * 验证结果
 */
bool verifyResults(const float* gpu_result, const float* cpu_result, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(gpu_result[i] - cpu_result[i]) > epsilon) {
            printf("❌ Mismatch at element %d: GPU = %.6f, CPU = %.6f\n", 
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}

/**
 * 初始化矩阵
 */
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = (float)(rand() % 10);
    }
}

/**
 * 正确性测试
 */
bool testCorrectness() {
    printf("\n=== 正确性测试 ===\n");
    
    // 测试非方阵：M(64×128) × N(128×96) = P(64×96)
    int m = 64, n = 128, o = 96;
    printf("测试: M(%d×%d) × N(%d×%d) = P(%d×%d)\n\n", m, n, n, o, m, o);
    
    // 分配内存
    float* h_M = (float*)malloc(m * n * sizeof(float));
    float* h_N = (float*)malloc(n * o * sizeof(float));
    float* h_P_gpu = (float*)malloc(m * o * sizeof(float));
    float* h_P_cpu = (float*)malloc(m * o * sizeof(float));
    
    // 初始化
    srand(2024);
    initializeMatrix(h_M, m * n);
    initializeMatrix(h_N, n * o);
    
    // GPU 计算
    printf("执行 GPU 计算...\n");
    matrixMultiplyDevice(h_P_gpu, h_M, h_N, m, n, o);
    
    // CPU 验证
    printf("执行 CPU 验证...\n");
    matrixMultiplyCPU(h_P_cpu, h_M, h_N, m, n, o);
    
    // 验证
    bool correct = verifyResults(h_P_gpu, h_P_cpu, m * o);
    printf("%s\n", correct ? "✅ 正确性测试通过" : "❌ 正确性测试失败");
    
    // 释放内存
    free(h_M);
    free(h_N);
    free(h_P_gpu);
    free(h_P_cpu);
    
    return correct;
}

/**
 * 性能测试
 */
void testPerformance() {
    printf("\n=== 性能测试 ===\n");
    
    // 方阵：M(512×512) × N(512×512)
    int size = 512;
    int m = size, n = size, o = size;
    int iterations = 20;
    
    printf("矩阵大小: %d×%d\n", size, size);
    printf("迭代次数: %d\n\n", iterations);
    
    // 分配内存
    float* h_M = (float*)malloc(m * n * sizeof(float));
    float* h_N = (float*)malloc(n * o * sizeof(float));
    float* h_P = (float*)malloc(m * o * sizeof(float));
    
    initializeMatrix(h_M, m * n);
    initializeMatrix(h_N, n * o);
    
    // 预热
    matrixMultiplyDevice(h_P, h_M, h_N, m, n, o);
    
    // 性能测试
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        matrixMultiplyDevice(h_P, h_M, h_N, m, n, o);
    }
    timer.stop();
    
    double avg_time = timer.elapsed_ms() / iterations;
    double operations = 2.0 * m * n * o;  // 浮点操作数
    double gflops = (operations / avg_time) / 1e6;
    
    printf("结果:\n");
    printf("  平均时间: %.3f ms\n", avg_time);
    printf("  性能: %.2f GFLOPS\n", gflops);
    
    // 释放内存
    free(h_M);
    free(h_N);
    free(h_P);
}

int main() {
    printf("==================================================\n");
    printf("第三章 - 练习3: 标准矩阵乘法（元素级 kernel)\n");
    printf("==================================================\n");
    
    printDeviceInfo();
    
    if (!testCorrectness()) {
        printf("\n❌ 正确性测试失败！\n");
        return 1;
    }
    
    testPerformance();
    
    printf("\n✅ 所有测试完成！\n\n");
    
    return 0;
}
