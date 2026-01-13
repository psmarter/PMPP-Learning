#include <stdio.h>
#include <cuda_runtime.h>
#include "../../../Common/utils.cuh"
#include "solution.h"

/**
 * CUDA Kernel: 矩阵乘法（行级）
 * 每个线程计算输出矩阵的一整行
 * P = M × N
 */
__global__ void matrixMulRowKernel(float* M, float* N, float* P, int size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size) {
        // 对该行的每个元素执行计算
        for (int col = 0; col < size; ++col) {
            float sum = 0.0f;
            for (int j = 0; j < size; ++j) {
                sum += M[row * size + j] * N[j * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}

/**
 * CUDA Kernel: 矩阵乘法（列级）
 * 每个线程计算输出矩阵的一整列
 * P = M × N
 */
__global__ void matrixMulColKernel(float* M, float* N, float* P, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < size) {
        // 对该列的每个元素执行计算
        for (int row = 0; row < size; ++row) {
            float sum = 0.0f;
            for (int j = 0; j < size; ++j) {
                sum += M[row * size + j] * N[j * size + col];
            }
            P[row * size + col] = sum;
        }
    }
}

/**
 * Host 函数：矩阵乘法（行级）的完整流程
 */
void matrixMultiplyRowDevice(float* h_P, const float* h_M, const float* h_N, int size) {
    // 1. 分配设备内存
    float *d_M, *d_N, *d_P;
    size_t bytes = size * size * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_M, bytes));
    CHECK_CUDA(cudaMalloc(&d_N, bytes));
    CHECK_CUDA(cudaMalloc(&d_P, bytes));
    
    // 2. 拷贝输入数据到设备
    CHECK_CUDA(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
    
    // 3. 配置并启动 kernel
    // 每个线程处理一行，所以需要 size 个线程
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    matrixMulRowKernel<<<gridSize, blockSize>>>(d_M, d_N, d_P, size);
    CHECK_LAST_CUDA_ERROR();
    
    // 4. 等待 GPU 完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost));
    
    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * Host 函数：矩阵乘法（列级）的完整流程
 */
void matrixMultiplyColDevice(float* h_P, const float* h_M, const float* h_N, int size) {
    // 1. 分配设备内存
    float *d_M, *d_N, *d_P;
    size_t bytes = size * size * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_M, bytes));
    CHECK_CUDA(cudaMalloc(&d_N, bytes));
    CHECK_CUDA(cudaMalloc(&d_P, bytes));
    
    // 2. 拷贝输入数据到设备
    CHECK_CUDA(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
    
    // 3. 配置并启动 kernel
    // 每个线程处理一列，所以需要 size 个线程
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    matrixMulColKernel<<<gridSize, blockSize>>>(d_M, d_N, d_P, size);
    CHECK_LAST_CUDA_ERROR();
    
    // 4. 等待 GPU 完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_P, d_P, bytes, cudaMemcpyDeviceToHost));
    
    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
