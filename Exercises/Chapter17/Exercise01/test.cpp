// ============================================================================
// test.cpp - 第十七章练习1: 共轭梯度法测试
// ============================================================================

#include "solution.h"
#include "../../../Common/timer.h"
#include "../../../Common/utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

const float EPSILON = 1e-4f;

// 生成对称正定矩阵 A = Q^T * D * Q
void generate_spd_matrix(float* A, int n) {
    // 使用简单方法：A = B^T * B + epsilon * I
    float* B = new float[n * n];
    
    for (int i = 0; i < n * n; i++) {
        B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
    
    // A = B^T * B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += B[k * n + i] * B[k * n + j];
            }
            A[i * n + j] = sum;
        }
    }
    
    // 添加对角增强确保正定
    for (int i = 0; i < n; i++) {
        A[i * n + i] += n * 0.1f;
    }
    
    delete[] B;
}

void generate_random_vector(float* v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

bool compare_vectors(const float* a, const float* b, int n, float epsilon = EPSILON) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    bool passed = max_diff <= epsilon;
    printf("  最大差异: %.2e %s\n", max_diff, passed ? "✅" : "❌");
    return passed;
}

// ============================================================================
// 测试小规模系统（验证算法正确性）
// ============================================================================

bool test_small_system() {
    printf("\n=== 小规模系统测试 (3x3) ===\n");
    
    // 使用参考仓库中的相同测试用例
    float A[] = {
        4, 1, 0,
        1, 3, 2,
        0, 2, 6
    };
    float b[] = {1, -2, 3};
    
    float x_cpu[3] = {0, 0, 0};
    float x_gpu[3] = {0, 0, 0};
    
    // CPU 求解
    int iter_cpu = cg_solve_cpu(A, b, x_cpu, 3, 1e-10f, 100);
    printf("CPU 迭代次数: %d\n", iter_cpu);
    printf("CPU 解: [%.6f, %.6f, %.6f]\n", x_cpu[0], x_cpu[1], x_cpu[2]);
    
    // 验证 A*x = b
    float Ax[3];
    matvec_multiply_cpu(A, x_cpu, Ax, 3);
    printf("A*x = [%.6f, %.6f, %.6f]\n", Ax[0], Ax[1], Ax[2]);
    printf("b   = [%.6f, %.6f, %.6f]\n", b[0], b[1], b[2]);
    
    float error = 0.0f;
    for (int i = 0; i < 3; i++) {
        error += (Ax[i] - b[i]) * (Ax[i] - b[i]);
    }
    error = sqrtf(error);
    printf("残差范数: %.2e\n", error);
    
    // GPU 求解
    float *d_A, *d_b, *d_x;
    cudaMalloc(&d_A, 9 * sizeof(float));
    cudaMalloc(&d_b, 3 * sizeof(float));
    cudaMalloc(&d_x, 3 * sizeof(float));
    
    cudaMemcpy(d_A, A, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, 3 * sizeof(float));
    
    int iter_gpu = cg_solve_gpu(d_A, d_b, d_x, 3, 1e-10f, 100);
    cudaMemcpy(x_gpu, d_x, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\nGPU 迭代次数: %d\n", iter_gpu);
    printf("GPU 解: [%.6f, %.6f, %.6f]\n", x_gpu[0], x_gpu[1], x_gpu[2]);
    
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    
    return compare_vectors(x_cpu, x_gpu, 3);
}

// ============================================================================
// 测试大规模系统（性能测试）
// ============================================================================

bool test_large_system() {
    printf("\n=== 大规模系统测试 (256x256) ===\n");
    
    int n = 256;
    float* A = new float[n * n];
    float* b = new float[n];
    float* x_cpu = new float[n];
    float* x_gpu = new float[n];
    
    generate_spd_matrix(A, n);
    generate_random_vector(b, n);
    
    // CPU 求解
    memset(x_cpu, 0, n * sizeof(float));
    Timer cpu_timer;
    cpu_timer.start();
    int iter_cpu = cg_solve_cpu(A, b, x_cpu, n, 1e-6f, 500);
    cpu_timer.stop();
    printf("CPU: %d 次迭代, %.2f ms\n", iter_cpu, cpu_timer.elapsed_ms());
    
    // GPU 求解
    float *d_A, *d_b, *d_x;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    
    cudaMemcpy(d_A, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, n * sizeof(float));
    
    CudaTimer gpu_timer;
    gpu_timer.start();
    int iter_gpu = cg_solve_gpu(d_A, d_b, d_x, n, 1e-6f, 500);
    gpu_timer.stop();
    
    cudaMemcpy(x_gpu, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU: %d 次迭代, %.2f ms\n", iter_gpu, gpu_timer.elapsed_ms());
    
    printf("加速比: %.2fx\n", cpu_timer.elapsed_ms() / gpu_timer.elapsed_ms());
    
    bool passed = compare_vectors(x_cpu, x_gpu, n, 1e-3f);
    
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    delete[] A;
    delete[] b;
    delete[] x_cpu;
    delete[] x_gpu;
    
    return passed;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  第十七章练习1：共轭梯度法 (Conjugate Gradient)\n");
    printf("  求解对称正定线性系统 Ax = b\n");
    printf("================================================================\n");
    
    srand(42);
    
    bool all_passed = true;
    all_passed &= test_small_system();
    all_passed &= test_large_system();
    
    printf("\n================================================================\n");
    if (all_passed) {
        printf("  ✅ 所有测试通过！\n");
    } else {
        printf("  ❌ 部分测试失败\n");
    }
    printf("================================================================\n");
    
    return all_passed ? 0 : 1;
}
