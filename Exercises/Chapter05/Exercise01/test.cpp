/**
 * 第五章：内存架构和数据局部性 - Tiled 矩阵乘法性能测试
 * 
 * 本程序对比朴素矩阵乘法和 Tiled 矩阵乘法的性能差异
 * 演示共享内存优化对性能的提升效果
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../../Common/timer.h"
#include "solution.h"

// 结果验证容差
const float TOLERANCE = 1e-3f;

/**
 * 初始化矩阵（随机值）
 */
void initMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

/**
 * CPU 参考实现（用于验证正确性）
 */
void matrixMulCPU(float* P, const float* M, const float* N, int m, int n, int o) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < o; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += M[i * n + k] * N[k * o + j];
            }
            P[i * o + j] = sum;
        }
    }
}

/**
 * 验证两个矩阵是否近似相等
 */
bool verifyResults(const float* A, const float* B, int size, float tolerance = TOLERANCE) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

/**
 * 基准测试函数（使用 Common/timer.h 中的 Timer 类）
 */
double benchmark(void (*func)(float*, const float*, const float*, int, int, int),
                 float* P, const float* M, const float* N, 
                 int m, int n, int o, int iterations) {
    // 预热
    func(P, M, N, m, n, o);
    
    Timer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        func(P, M, N, m, n, o);
    }
    timer.stop();
    
    return timer.elapsed_ms() / iterations;
}

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  第五章：内存架构和数据局部性\n");
    printf("  Tiled Matrix Multiplication Benchmark\n");
    printf("========================================\n\n");

    // 矩阵大小
    const int M_ROWS = 1024;
    const int M_COLS = 1024;
    const int N_COLS = 1024;
    const int ITERATIONS = 10;

    printf("矩阵大小: %d × %d × %d\n", M_ROWS, M_COLS, N_COLS);
    printf("测试迭代次数: %d\n\n", ITERATIONS);

    // 分配内存
    float* h_M = new float[M_ROWS * M_COLS];
    float* h_N = new float[M_COLS * N_COLS];
    float* h_P_naive = new float[M_ROWS * N_COLS];
    float* h_P_tiled = new float[M_ROWS * N_COLS];

    // 初始化矩阵
    srand(42);
    initMatrix(h_M, M_ROWS, M_COLS);
    initMatrix(h_N, M_COLS, N_COLS);

    printf("=== 正确性验证 ===\n");
    
    // 运行朴素版本
    matrixMul(h_P_naive, h_M, h_N, M_ROWS, M_COLS, N_COLS);
    
    // 运行 Tiled 版本
    matrixMulTiled(h_P_tiled, h_M, h_N, M_ROWS, M_COLS, N_COLS);
    
    // 验证结果
    if (verifyResults(h_P_naive, h_P_tiled, M_ROWS * N_COLS)) {
        printf("✅ 两种方法结果一致！\n\n");
    } else {
        printf("❌ 结果不一致！\n\n");
    }

    printf("=== 性能测试 ===\n");
    
    // 朴素版本性能
    double naiveTime = benchmark(matrixMul, h_P_naive, h_M, h_N, 
                                  M_ROWS, M_COLS, N_COLS, ITERATIONS);
    printf("朴素矩阵乘法:    %.3f ms\n", naiveTime);
    
    // Tiled 版本性能
    double tiledTime = benchmark(matrixMulTiled, h_P_tiled, h_M, h_N, 
                                  M_ROWS, M_COLS, N_COLS, ITERATIONS);
    printf("Tiled 矩阵乘法:  %.3f ms\n", tiledTime);
    
    // 计算加速比
    double speedup = naiveTime / tiledTime;
    printf("\n加速比: %.2fx\n", speedup);

    // 计算 GFLOPS
    double gflops = 2.0 * M_ROWS * M_COLS * N_COLS / 1e9;
    printf("\n计算量: %.2f GFLOP\n", gflops);
    printf("朴素版本吞吐量:  %.2f GFLOPS\n", gflops / (naiveTime / 1000.0));
    printf("Tiled 版本吞吐量: %.2f GFLOPS\n", gflops / (tiledTime / 1000.0));

    printf("\n");
    printf("【关键概念】\n");
    printf("----------------------------------------\n");
    printf("• Tile 大小: 32×32\n");
    printf("• 共享内存: 每块 2×32×32×4 = 8 KB\n");
    printf("• 全局内存访问减少: 约 32 倍\n");
    printf("• 算术强度提升: 0.25 → 8 FLOP/Byte\n");
    printf("\n");

    // 清理
    delete[] h_M;
    delete[] h_N;
    delete[] h_P_naive;
    delete[] h_P_tiled;

    printf("✅ 测试完成！\n\n");
    return 0;
}
