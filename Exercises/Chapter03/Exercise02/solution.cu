#include <stdio.h>
#include <cuda_runtime.h>
#include "../../../Common/utils.cuh"
#include "solution.h"

/**
 * CUDA Kernel: 矩阵向量乘法
 * A[i] = sum_over_j(B[i][j] * C[j])
 * 每个线程计算输出向量的一个元素
 */
__global__ void matrixVecMulKernel(float* B, float* C, float* A, int matrix_rows, int matrix_cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < matrix_rows) {
        float sum = 0.0f;
        for (int j = 0; j < matrix_cols; ++j) {
            sum += B[i * matrix_cols + j] * C[j];
        }
        A[i] = sum;
    }
}

/**
 * Host 函数：矩阵向量乘法的完整流程
 */
void matrixVectorMultiplyDevice(float* h_A, const float* h_B, const float* h_C, 
                               int matrix_rows, int matrix_cols) {
    // 1. 分配设备内存
    float *d_A, *d_B, *d_C;
    size_t matrix_bytes = matrix_rows * matrix_cols * sizeof(float);
    size_t vector_bytes = matrix_cols * sizeof(float);
    size_t result_bytes = matrix_rows * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_A, result_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, vector_bytes));
    
    // 2. 拷贝输入数据到设备
    CHECK_CUDA(cudaMemcpy(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, vector_bytes, cudaMemcpyHostToDevice));
    
    // 3. 配置并启动 kernel
    // 每个线程计算一个输出元素，需要 matrix_rows 个线程
    int blockSize = 256;
    int gridSize = (matrix_rows + blockSize - 1) / blockSize;
    
    matrixVecMulKernel<<<gridSize, blockSize>>>(d_B, d_C, d_A, matrix_rows, matrix_cols);
    CHECK_LAST_CUDA_ERROR();
    
    // 4. 等待 GPU 完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_A, d_A, result_bytes, cudaMemcpyDeviceToHost));
    
    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}
