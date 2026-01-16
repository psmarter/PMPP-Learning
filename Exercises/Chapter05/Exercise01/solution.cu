#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Tile 宽度（共享内存块大小）
#define TILE_WIDTH 32

/**
 * 朴素矩阵乘法 Kernel
 * 每个线程计算输出矩阵的一个元素
 * 直接从全局内存读取所需数据（无优化）
 */
__global__ void MatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < o) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += M[row * n + i] * N[i * o + col];
        }
        P[row * o + col] = sum;
    }
}

/**
 * Tiled 矩阵乘法 Kernel
 * 使用共享内存分块加载数据，减少全局内存访问
 * 
 * 关键优化：
 * 1. 将 M 和 N 的 Tile 加载到共享内存
 * 2. Block 内线程协作加载（每个线程加载一个元素）
 * 3. 在共享内存上计算部分点积
 * 4. 累加所有 Tile 的结果
 */
__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    // 声明共享内存（Block 内所有线程共享）
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算该线程负责的 P 元素位置
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // 分块循环：遍历所有 Tile
    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < numTiles; ++ph) {
        
        // 协作加载 M 的 Tile（带边界检查）
        int mCol = ph * TILE_WIDTH + tx;
        if (row < m && mCol < n) {
            Mds[ty][tx] = M[row * n + mCol];
        } else {
            Mds[ty][tx] = 0.0f;  // 越界填 0
        }

        // 协作加载 N 的 Tile（带边界检查）
        int nRow = ph * TILE_WIDTH + ty;
        if (nRow < n && col < o) {
            Nds[ty][tx] = N[nRow * o + col];
        } else {
            Nds[ty][tx] = 0.0f;  // 越界填 0
        }

        // 第一次同步：确保 Tile 加载完成
        __syncthreads();

        // 使用共享内存计算部分点积
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        // 第二次同步：确保计算完成再加载下一个 Tile
        __syncthreads();
    }

    // 写入结果
    if (row < m && col < o) {
        P[row * o + col] = Pvalue;
    }
}

/**
 * 朴素矩阵乘法（主机接口函数）
 */
void matrixMul(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    // 配置 kernel 启动参数
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, 
                 (m + dimBlock.y - 1) / dimBlock.y);

    // 启动 kernel
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * Tiled 矩阵乘法（主机接口函数）
 */
void matrixMulTiled(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    // 复制数据到设备
    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    // 配置 kernel 启动参数
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((o + dimBlock.x - 1) / dimBlock.x, 
                 (m + dimBlock.y - 1) / dimBlock.y);

    // 启动 Tiled kernel
    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    // 复制结果回主机
    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    // 释放设备内存
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
