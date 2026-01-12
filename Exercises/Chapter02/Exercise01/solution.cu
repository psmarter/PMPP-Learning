#include <stdio.h>
#include <cuda_runtime.h>
#include "../../../Common/utils.cuh"

/**
 * CUDA Kernel: 向量乘法
 * 每个线程计算一个元素：C[i] = A[i] * B[i]
 */
__global__ void vectorMultiply(float* c, const float* a, const float* b, int n) {
    // 计算当前线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查：确保不会越界访问
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

/**
 * Host 函数：向量乘法的完整流程
 */
void vectorMultiplyDevice(float* h_c, const float* h_a, const float* h_b, int n) {
    // 1. 分配设备内存
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    
    // 2. 拷贝输入数据到设备
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // 3. 配置并启动 kernel
    int blockSize = 256;  // 每个块 256 个线程
    int gridSize = (n + blockSize - 1) / blockSize;  // 向上取整
    
    vectorMultiply<<<gridSize, blockSize>>>(d_c, d_a, d_b, n);
    CHECK_LAST_CUDA_ERROR();
    
    // 4. 等待 GPU 完成
    CHECK_CUDA(cudaDeviceSynchronize());       
    
    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));         // 这个会自动等待
    
    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}
