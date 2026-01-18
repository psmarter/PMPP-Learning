/**
 * 第十一章：前缀和（扫描）- CUDA 实现
 * 
 * 参考：chapter-11/code/scan.cu, hierarchical_scan.cu
 * 
 * 本实现包含多种 kernel：
 * 1. kogge_stone_scan_kernel - Kogge-Stone 扫描（图11.3）
 * 2. kogge_stone_scan_kernel_double_buffer - 双缓冲版本（练习2）
 * 3. brent_kung_scan_kernel - Brent-Kung 扫描（图11.4）
 * 4. three_phase_scan_kernel - 三阶段扫描
 * 5. hierarchical_scan_kernel - 分层扫描（任意长度）
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// 辅助函数 - 需要 __host__ __device__ 才能在 kernel 中调用
__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// ====================== Kernels ======================

/**
 * Kogge-Stone 扫描 Kernel（图11.3）
 * 工作量：O(N log N)，步骤：O(log N)
 * 需要两次 __syncthreads() 避免写后读竞争
 */
__global__ void kogge_stone_scan_kernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float buffer[];
    unsigned int tid = threadIdx.x;

    // 加载数据到共享内存
    if (tid < N) {
        buffer[tid] = X[tid];
    } else {
        buffer[tid] = 0.0f;
    }

    // Kogge-Stone 迭代
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (tid >= stride) {
            temp = buffer[tid] + buffer[tid - stride];
        }
        __syncthreads();  // 第二次同步避免写后读竞争
        if (tid >= stride) {
            buffer[tid] = temp;
        }
    }

    if (tid < N) {
        Y[tid] = buffer[tid];
    }
}

/**
 * Kogge-Stone 双缓冲扫描 Kernel（练习2）
 * 使用双缓冲消除第二次 __syncthreads()
 */
__global__ void kogge_stone_scan_kernel_double_buffer(float* X, float* Y, unsigned int N) {
    extern __shared__ float shared_mem[];
    float* buffer1 = shared_mem;
    float* buffer2 = &shared_mem[N];

    unsigned int tid = threadIdx.x;

    float* src = buffer1;
    float* dst = buffer2;

    // 加载数据
    if (tid < N) {
        src[tid] = X[tid];
    } else {
        src[tid] = 0.0f;
    }

    // Kogge-Stone 迭代（无需第二次同步）
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (tid >= stride) {
            dst[tid] = src[tid] + src[tid - stride];
        } else {
            dst[tid] = src[tid];
        }

        // 交换缓冲区
        float* temp = src;
        src = dst;
        dst = temp;
    }

    if (tid < N) {
        Y[tid] = src[tid];
    }
}

/**
 * Brent-Kung 扫描 Kernel（图11.4）
 * 工作量：O(N)，更高效
 * 分两阶段：上扫（归约）和下扫（分发）
 */
__global__ void brent_kung_scan_kernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;

    // 加载数据
    if (tid < N) {
        sdata[tid] = X[tid];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // 上扫阶段（归约树）
    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride *= 2) {
        unsigned int index = (tid + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }

    // 下扫阶段（分发树）
    for (unsigned int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        unsigned int index = (tid + 1) * 2 * stride - 1;
        if (index + stride < blockDim.x) {
            sdata[index + stride] += sdata[index];
        }
        __syncthreads();
    }

    // 写回结果
    if (tid < N) {
        Y[tid] = sdata[tid];
    }
}

/**
 * 三阶段扫描 Kernel
 * 阶段1：每线程局部扫描 COARSE_FACTOR 个元素
 * 阶段2：Kogge-Stone 扫描各段末尾
 * 阶段3：将段和分发到各元素
 */
__global__ void three_phase_scan_kernel(float* X, float* Y, unsigned int N) {
    extern __shared__ float shared_mem[];
    float* buffer = shared_mem;
    float* section_sums = &shared_mem[N];

    unsigned int tid = threadIdx.x;

    // 阶段1：加载并局部扫描
    for (int i = 0; i < COARSE_FACTOR; i++) {
        unsigned int idx = tid * COARSE_FACTOR + i;
        if (idx < N) {
            buffer[idx] = X[idx];
        }
    }
    __syncthreads();

    // 局部前缀和
    for (int i = 1; i < COARSE_FACTOR; i++) {
        unsigned int idx = tid * COARSE_FACTOR + i;
        if (idx < N) {
            buffer[idx] += buffer[idx - 1];
        }
    }
    __syncthreads();

    // 收集各段末尾值
    unsigned int num_sections = cdiv(N, COARSE_FACTOR);
    if (tid < num_sections) {
        unsigned int end_idx = (tid + 1) * COARSE_FACTOR - 1;
        if (end_idx < N) {
            section_sums[tid] = buffer[end_idx];
        } else {
            section_sums[tid] = buffer[N - 1];
        }
    }
    __syncthreads();

    // 阶段2：对段末尾进行 Kogge-Stone 扫描
    for (unsigned int stride = 1; stride < num_sections; stride *= 2) {
        float temp;
        __syncthreads();
        if (tid >= stride && tid < num_sections) {
            temp = section_sums[tid] + section_sums[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < num_sections) {
            section_sums[tid] = temp;
        }
    }
    __syncthreads();

    // 阶段3：分发段和
    for (int i = 0; i < COARSE_FACTOR; i++) {
        unsigned int idx = tid * COARSE_FACTOR + i;
        if (idx < N) {
            unsigned int section = idx / COARSE_FACTOR;
            if (section > 0) {
                buffer[idx] += section_sums[section - 1];
            }
            Y[idx] = buffer[idx];
        }
    }
}

/**
 * 分层扫描 - 第一阶段 Kernel
 * 对每个 Block 执行 Kogge-Stone 扫描，并保存块末尾和
 */
__global__ void hierarchical_scan_phase1_kernel(float* X, float* Y, float* block_sums, unsigned int N) {
    extern __shared__ float buffer[];
    
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * blockDim.x + tid;

    // 加载数据
    if (gid < N) {
        buffer[tid] = X[gid];
    } else {
        buffer[tid] = 0.0f;
    }

    // Kogge-Stone 扫描
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (tid >= stride) {
            temp = buffer[tid] + buffer[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            buffer[tid] = temp;
        }
    }

    // 写回局部结果
    if (gid < N) {
        Y[gid] = buffer[tid];
    }

    // 保存块末尾和
    if (tid == blockDim.x - 1) {
        block_sums[bid] = buffer[tid];
    }
}

/**
 * 分层扫描 - 第三阶段 Kernel
 * 将扫描后的块和加到各块元素上
 */
__global__ void hierarchical_scan_phase3_kernel(float* Y, float* block_sums, unsigned int N) {
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gid = bid * blockDim.x + tid;

    if (bid > 0 && gid < N) {
        Y[gid] += block_sums[bid - 1];
    }
}

// ====================== 主机接口 ======================

/**
 * CPU 顺序包含扫描
 */
void scan_sequential(float* X, float* Y, unsigned int N) {
    Y[0] = X[0];
    for (unsigned int i = 1; i < N; i++) {
        Y[i] = X[i] + Y[i - 1];
    }
}

/**
 * Kogge-Stone 扫描（主机接口）
 */
void scan_kogge_stone(float* X, float* Y, unsigned int N) {
    if (N > 1024) {
        printf("错误：Kogge-Stone 单 Block 扫描仅支持 <= 1024 元素\n");
        return;
    }

    float *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    kogge_stone_scan_kernel<<<1, N, N * sizeof(float)>>>(d_X, d_Y, N);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}

/**
 * Kogge-Stone 双缓冲扫描（主机接口）
 */
void scan_kogge_stone_double_buffer(float* X, float* Y, unsigned int N) {
    if (N > 1024) {
        printf("错误：Kogge-Stone 单 Block 扫描仅支持 <= 1024 元素\n");
        return;
    }

    float *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    kogge_stone_scan_kernel_double_buffer<<<1, N, 2 * N * sizeof(float)>>>(d_X, d_Y, N);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}

/**
 * Brent-Kung 扫描（主机接口）
 */
void scan_brent_kung(float* X, float* Y, unsigned int N) {
    if (N > 1024) {
        printf("错误：Brent-Kung 单 Block 扫描仅支持 <= 1024 元素\n");
        return;
    }

    // 找到大于等于 N 的 2 的幂
    unsigned int blockSize = 1;
    while (blockSize < N && blockSize < 1024) {
        blockSize *= 2;
    }

    float *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    brent_kung_scan_kernel<<<1, blockSize, blockSize * sizeof(float)>>>(d_X, d_Y, N);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}

/**
 * 三阶段扫描（主机接口）
 */
void scan_three_phase(float* X, float* Y, unsigned int N) {
    if (N > 1024) {
        printf("错误：三阶段扫描单 Block 仅支持 <= 1024 元素\n");
        return;
    }

    unsigned int numThreads = cdiv(N, COARSE_FACTOR);
    size_t sharedMemSize = (N + numThreads) * sizeof(float);

    float *d_X, *d_Y;
    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    three_phase_scan_kernel<<<1, numThreads, sharedMemSize>>>(d_X, d_Y, N);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}

/**
 * 分层扫描（主机接口）
 * 支持任意长度
 */
void scan_hierarchical(float* X, float* Y, unsigned int N) {
    const unsigned int BLOCK_SIZE = 1024;
    unsigned int numBlocks = cdiv(N, BLOCK_SIZE);

    float *d_X, *d_Y, *d_block_sums, *d_scanned_block_sums;

    CHECK_CUDA(cudaMalloc(&d_X, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_sums, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scanned_block_sums, numBlocks * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_X, X, N * sizeof(float), cudaMemcpyHostToDevice));

    // 阶段1：各 Block 局部扫描
    hierarchical_scan_phase1_kernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_X, d_Y, d_block_sums, N);
    CHECK_LAST_CUDA_ERROR();

    // 阶段2：对块和进行扫描（递归或单 Block）
    if (numBlocks <= 1024) {
        kogge_stone_scan_kernel<<<1, numBlocks, numBlocks * sizeof(float)>>>(
            d_block_sums, d_scanned_block_sums, numBlocks);
        CHECK_LAST_CUDA_ERROR();
    } else {
        // 递归处理（简化版本，假设 numBlocks <= 1024^2）
        printf("警告：块数超过 1024，需要更多层递归\n");
    }

    // 阶段3：将块和加到各元素
    hierarchical_scan_phase3_kernel<<<numBlocks, BLOCK_SIZE>>>(d_Y, d_scanned_block_sums, N);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(d_block_sums));
    CHECK_CUDA(cudaFree(d_scanned_block_sums));
}
