/**
 * 第九章：并行直方图 - CUDA 实现
 * 
 * 参考：chapter-09/code/histogram.cu
 * 
 * 本实现包含5种 kernel：
 * 1. histogram_sequential - CPU顺序实现
 * 2. histo_kernel - 基础并行（图9.6）
 * 3. histo_private_kernel_shared_memory - 私有化共享内存（图9.10）
 * 4. histo_private_kernel_thread_coarsening - 线程粗化（图9.14）
 * 5. histo_private_kernel_coalesced - 线程粗化 + 内存合并
 */

#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// 辅助函数：向上取整除法
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

// 获取 SM 数量（用于内存合并访问优化）
int getNumSMs() {
    static int numSMs = 0;
    if (numSMs == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        numSMs = prop.multiProcessorCount;
    }
    return numSMs;
}

/**
 * 打印直方图内容
 */
void print_histogram(const unsigned int* histogram, const char* label) {
    printf("\n%s:\n", label);
    for (int i = 0; i < NUM_BINS; i++) {
        char start = 'a' + (i * BIN_SIZE);
        char end = 'a' + ((i + 1) * BIN_SIZE - 1);
        if (end > 'z') end = 'z';
        printf("  Bin %d (%c-%c): %u\n", i, start, end, histogram[i]);
    }
}

/**
 * CPU 顺序实现
 */
void histogram_sequential(char* data, unsigned int length, unsigned int* histogram) {
    // 初始化
    for (int i = 0; i < NUM_BINS; i++) {
        histogram[i] = 0;
    }
    // 统计
    for (unsigned int i = 0; i < length; ++i) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            histogram[alphabet_position / BIN_SIZE]++;
        }
    }
}

/**
 * 基础并行 Kernel（图9.6）
 * 每个线程处理一个元素，直接对全局内存进行原子操作
 * 问题：高度争用，性能较差
 */
__global__ void histo_kernel(char* data, unsigned int length, unsigned int* histo) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo[alphabet_position / BIN_SIZE], 1);
        }
    }
}

/**
 * 基础并行（主机接口）
 */
void histogram_parallel_basic(char* data, unsigned int length, unsigned int* histogram) {
    char* d_data;
    unsigned int* d_histo;

    CHECK_CUDA(cudaMalloc((void**)&d_data, length * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(cdiv(length, dimBlock.x));

    histo_kernel<<<dimGrid, dimBlock>>>(d_data, length, d_histo);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(histogram, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));
}

/**
 * 私有化共享内存 Kernel（图9.10）
 * 每个 Block 使用共享内存维护私有直方图
 * 优点：共享内存原子操作比全局内存快得多
 */
__global__ void histo_private_kernel_shared_memory(char* data, unsigned int length, unsigned int* histo) {
    // 初始化私有直方图（共享内存）
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // 统计到私有直方图
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position / BIN_SIZE], 1);
        }
    }
    __syncthreads();

    // 合并到全局直方图
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}

/**
 * 私有化共享内存（主机接口）
 */
void histogram_parallel_private_shared(char* data, unsigned int length, unsigned int* histogram) {
    char* d_data;
    unsigned int* d_histo;

    CHECK_CUDA(cudaMalloc((void**)&d_data, length * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(cdiv(length, dimBlock.x));

    histo_private_kernel_shared_memory<<<dimGrid, dimBlock>>>(d_data, length, d_histo);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(histogram, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));
}

/**
 * 线程粗化 Kernel（图9.14）
 * 每个线程处理连续的 COARSEN_FACTOR 个元素
 * 优点：减少 Block 数量，减少合并阶段的全局原子操作
 */
__global__ void histo_private_kernel_thread_coarsening(char* data, unsigned int length, unsigned int* histo) {
    // 初始化私有直方图
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // 每个线程处理 COARSEN_FACTOR 个连续元素
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start = tid * COARSEN_FACTOR;
    unsigned int end = min(start + COARSEN_FACTOR, length);
    
    for (unsigned int i = start; i < end; ++i) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position / BIN_SIZE], 1);
        }
    }
    __syncthreads();

    // 合并到全局直方图
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}

/**
 * 线程粗化（主机接口）
 */
void histogram_parallel_coarsening(char* data, unsigned int length, unsigned int* histogram) {
    char* d_data;
    unsigned int* d_histo;

    CHECK_CUDA(cudaMalloc((void**)&d_data, length * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));

    // 由于线程粗化，每个线程处理 COARSEN_FACTOR 个元素
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(cdiv(length, dimBlock.x * COARSEN_FACTOR));

    histo_private_kernel_thread_coarsening<<<dimGrid, dimBlock>>>(d_data, length, d_histo);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(histogram, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));
}

/**
 * 线程粗化 + 内存合并访问 Kernel
 * 使用 Grid-Stride Loop 实现更好的内存访问模式
 * 相邻线程访问相邻内存位置，实现内存合并
 */
__global__ void histo_private_kernel_coalesced(char* data, unsigned int length, unsigned int* histo) {
    // 初始化私有直方图
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Grid-Stride Loop：相邻线程访问相邻内存
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    for (unsigned int i = tid; i < length; i += stride) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position / BIN_SIZE], 1);
        }
    }
    __syncthreads();

    // 合并到全局直方图
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if (binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}

/**
 * 线程粗化 + 内存合并访问（主机接口）
 */
void histogram_parallel_coalesced(char* data, unsigned int length, unsigned int* histogram) {
    char* d_data;
    unsigned int* d_histo;

    CHECK_CUDA(cudaMalloc((void**)&d_data, length * sizeof(char)));
    CHECK_CUDA(cudaMemcpy(d_data, data, length * sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int)));

    // 固定 Grid 大小，每个 SM 运行多个 Block
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(getNumSMs() * 32);  // 每个 SM 32 个 Block

    histo_private_kernel_coalesced<<<dimGrid, dimBlock>>>(d_data, length, d_histo);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(histogram, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histo));
}
