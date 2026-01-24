#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Tile 宽度（共享内存块大小）
#define TILE_WIDTH 32

// 线程粗化因子：每个线程计算 COARSE_FACTOR 个输出元素
#define COARSE_FACTOR 4

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
 */
__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < numTiles; ++ph) {
        
        int mCol = ph * TILE_WIDTH + tx;
        if (row < m && mCol < n) {
            Mds[ty][tx] = M[row * n + mCol];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        int nRow = ph * TILE_WIDTH + ty;
        if (nRow < n && col < o) {
            Nds[ty][tx] = N[nRow * o + col];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if (row < m && col < o) {
        P[row * o + col] = Pvalue;
    }
}

/**
 * Tiled + Thread Coarsening 矩阵乘法 Kernel
 * 
 * 核心优化思想：
 * 1. 每个线程计算 COARSE_FACTOR 个输出元素（水平方向）
 * 2. M 矩阵的 Tile 被复用 COARSE_FACTOR 次
 * 3. 减少了 M 矩阵 Tile 的加载次数
 * 
 * 算术强度从 8 OP/B (Tiled) 提升到约 12.8 OP/B
 */
__global__ void TiledMatrixMulKernelWithCoarsening(float* M, float* N, float* P, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // 当前线程负责的行
    int row = by * TILE_WIDTH + ty;
    // 当前线程负责的起始列（每个线程处理 COARSE_FACTOR 个列）
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // COARSE_FACTOR 个输出值
    float Pvalue[COARSE_FACTOR] = {0.0f};

    int numTiles = (n + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < numTiles; ++ph) {
        
        // 加载 M 的 Tile（只需加载一次，复用 COARSE_FACTOR 次）
        int mCol = ph * TILE_WIDTH + tx;
        if (row < m && mCol < n) {
            Mds[ty][tx] = M[row * n + mCol];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        // 对每个粗化的列，加载对应的 N 的 Tile 并计算
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c * COARSE_FACTOR;

            // 加载 N 的 Tile
            int nRow = ph * TILE_WIDTH + ty;
            if (nRow < n && col < o) {
                Nds[ty][tx] = N[nRow * o + col];
            } else {
                Nds[ty][tx] = 0.0f;
            }

            __syncthreads();

            // 使用共享内存计算部分点积
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }

            __syncthreads();
        }
    }

    // 写入 COARSE_FACTOR 个结果
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * COARSE_FACTOR;
        if (row < m && col < o) {
            P[row * o + col] = Pvalue[c];
        }
    }
}

// 辅助函数：向上取整除法
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

/**
 * 朴素矩阵乘法（主机接口函数）
 */
void matrixMul(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));

    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * Tiled 矩阵乘法（主机接口函数）
 */
void matrixMulTiled(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(cdiv(o, dimBlock.x), cdiv(m, dimBlock.y));

    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}

/**
 * Tiled + Thread Coarsening 矩阵乘法（主机接口函数）
 */
void matrixMulTiledCoarsened(float* h_P, const float* h_M, const float* h_N, int m, int n, int o) {
    float *d_M, *d_N, *d_P;

    CHECK_CUDA(cudaMalloc(&d_M, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_N, n * o * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, m * o * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_M, h_M, m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, n * o * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    // Grid 宽度减小 COARSE_FACTOR 倍（因为每个线程处理更多列）
    dim3 dimGrid(cdiv(o, dimBlock.x * COARSE_FACTOR), cdiv(m, dimBlock.y));

    TiledMatrixMulKernelWithCoarsening<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, m, n, o);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_P, d_P, m * o * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
}
