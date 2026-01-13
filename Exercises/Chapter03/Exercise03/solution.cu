#include <stdio.h>
#include <cuda_runtime.h>
#include "../../../Common/utils.cuh"
#include "solution.h"

/**
 * CUDA Kernel: 标准矩阵乘法（元素级）
 * 每个线程计算输出矩阵的一个元素
 * P[row][col] = sum(M[row][k] * N[k][col])
 * 
 * 这是最常用的矩阵乘法实现方式，充分利用 GPU 的并行性
 */
__global__ void MatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < m && col < o) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            // M: (m × n); N: (n × o)
            sum += M[row * n + k] * N[k * o + col];
        }
        // P: (m × o)
        P[row * o + col] = sum;
    }
}

/**
 * Host 函数：标准矩阵乘法的完整流程
 */
void matrixMultiplyDevice(float* h_P, const float* h_M, const float* h_N, 
                         int m, int n, int o) {
    // 1. 分配设备内存
    float *d_M, *d_N, *d_P;
    size_t M_bytes = m * n * sizeof(float);
    size_t N_bytes = n * o * sizeof(float);
    size_t P_bytes = m * o * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_M, M_bytes));
    CHECK_CUDA(cudaMalloc(&d_N, N_bytes));
    CHECK_CUDA(cudaMalloc(&d_P, P_bytes));
    
    // 2. 拷贝输入数据到设备
    CHECK_CUDA(cudaMemcpy(d_M, h_M, M_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, N_bytes, cudaMemcpyHostToDevice));
    
    // 3. 配置并启动 kernel
    // 使用 2D 线程块和网格
    dim3 blockDim(16, 16);  // 16×16 = 256 个线程per块
    dim3 gridDim((o + blockDim.x - 1) / blockDim.x, 
                 (m + blockDim.y - 1) / blockDim.y);
    
    MatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    
    // 4. 等待 GPU 完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_P, d_P, P_bytes, cudaMemcpyDeviceToHost));
    
    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
