#include <iostream>
#include <cmath>
#include <cstdlib>
#include "../../../Common/timer.h"
#include "../../../Common/utils.cuh"
#include "solution.h"

/**
 * CPU 实现的向量乘法（用于验证）
 */
void vectorMultiplyCPU(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

/**
 * 验证结果正确性
 */
bool verifyResults(const float* gpu_result, const float* cpu_result, int n, float epsilon = 1e-5) {
    for (int i = 0; i < n; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > epsilon) {
            std::cerr << "❌ Mismatch at index " << i << ": "
                      << "GPU = " << gpu_result[i] << ", "
                      << "CPU = " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * 正确性测试
 */
bool testCorrectness() {
    const int N = 1 << 20;  // 1M 元素
    printf("\n=== Correctness Test ===\n");
    printf("Testing vector multiplication with %d elements...\n", N);
    
    // 分配主机内存
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c_gpu = new float[N];
    float *h_c_cpu = new float[N];
    
    // 初始化输入数据
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // GPU 计算
    vectorMultiplyDevice(h_c_gpu, h_a, h_b, N);
    
    // CPU 计算（参考结果）
    vectorMultiplyCPU(h_c_cpu, h_a, h_b, N);
    
    // 验证结果
    bool passed = verifyResults(h_c_gpu, h_c_cpu, N);
    
    if (passed) {
        std::cout << "✅ Correctness test PASSED!" << std::endl;
    } else {
        std::cout << "❌ Correctness test FAILED!" << std::endl;
    }
    
    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_gpu;
    delete[] h_c_cpu;
    
    return passed;
}

/**
 * 性能测试
 */
void testPerformance() {
    const int N = 1 << 20;  // 1M 元素
    const int iterations = 100;
    
    printf("\n=== Performance Test ===\n");
    printf("Data size: %d elements (%.2f MB)\n", N, N * sizeof(float) / 1024.0 / 1024.0);
    printf("Iterations: %d\n", iterations);
    
    // 分配主机内存
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // 预热
    vectorMultiplyDevice(h_c, h_a, h_b, N);
    
    // 性能测试
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        vectorMultiplyDevice(h_c, h_a, h_b, N);
    }
    timer.stop();
    
    // 计算平均时间
    float avgTime = timer.elapsed_ms() / iterations;
    
    // 计算有效带宽
    // 向量乘法：读取 A 和 B（2次读），写入 C（1次写），共 3 次内存访问
    float bytesAccessed = 3.0f * N * sizeof(float);
    float bandwidth = (bytesAccessed / avgTime) / 1e6;  // GB/s
    
    printf("\nResults:\n");
    printf("  Average time per iteration: %.3f ms\n", avgTime);
    printf("  Effective bandwidth: %.2f GB/s\n", bandwidth);
    
    // 释放内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

int main() {
    // 打印设备信息
    printDeviceInfo();
    
    // 运行正确性测试
    if (!testCorrectness()) {
        return 1;
    }
    
    // 运行性能测试
    testPerformance();
    
    printf("\n✅ All tests completed successfully!\n");
    return 0;
}
