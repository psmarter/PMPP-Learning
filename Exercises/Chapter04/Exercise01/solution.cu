#include "solution.h"
#include "../../../Common/utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

/**
 * 打印所有 CUDA 设备的详细属性
 * 
 * 本函数用于了解 GPU 的硬件规格，这对于：
 * - 优化 kernel 配置（块大小、网格大小）
 * - 计算占用率
 * - 理解资源限制（寄存器、共享内存）
 * 非常重要。
 * 
 * 注意：Common/utils.cuh 中提供了简化版的 printDeviceInfo()，
 * 这里提供了更详细的中文版本。
 */
void printDeviceProperties() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount 返回错误: %d\n-> %s\n", 
               (int)error, cudaGetErrorString(error));
        printf("结果 = 失败\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("未检测到支持 CUDA 的设备\n");
        exit(EXIT_FAILURE);
    }

    printf("检测到 %d 个 CUDA 设备\n", deviceCount);
    printf("=" );
    for (int i = 0; i < 50; i++) printf("=");
    printf("\n");

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("\n");
        printf("设备 %d: \"%s\"\n", dev, prop.name);
        printf("-");
        for (int i = 0; i < 50; i++) printf("-");
        printf("\n");

        // 计算能力
        printf("  计算能力:                      %d.%d\n", prop.major, prop.minor);
        
        // 内存信息
        printf("\n  【内存信息】\n");
        printf("  全局内存总量:                  %.2f GB\n", 
               (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  常量内存总量:                  %zu bytes (%.2f KB)\n", 
               prop.totalConstMem, prop.totalConstMem / 1024.0f);
        printf("  每块共享内存:                  %zu bytes (%.2f KB)\n", 
               prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0f);
        printf("  每块寄存器数量:                %d\n", prop.regsPerBlock);
        printf("  L2 缓存大小:                   %d bytes (%.2f MB)\n", 
               prop.l2CacheSize, prop.l2CacheSize / (1024.0f * 1024.0f));

        // SM 信息
        printf("\n  【SM (流式多处理器) 信息】\n");
        printf("  多处理器 (SM) 数量:            %d\n", prop.multiProcessorCount);
        printf("  每个 SM 最大线程数:            %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  每个 SM 最大线程块数:          %d\n", prop.maxBlocksPerMultiProcessor);

        // Warp 和线程信息
        printf("\n  【线程和 Warp 信息】\n");
        printf("  Warp 大小:                     %d\n", prop.warpSize);
        printf("  每块最大线程数:                %d\n", prop.maxThreadsPerBlock);
        printf("  每个块各维度最大线程数:        %d x %d x %d\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

        // 网格信息
        printf("\n  【网格信息】\n");
        printf("  每个网格各维度最大块数:        %d x %d x %d\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // 时钟和带宽
        printf("\n  【时钟和带宽】\n");
        printf("  GPU 时钟频率:                  %.2f GHz\n", prop.clockRate * 1e-6f);
        printf("  内存时钟频率:                  %.2f GHz\n", prop.memoryClockRate * 1e-6f);
        printf("  内存总线宽度:                  %d-bit\n", prop.memoryBusWidth);
        
        // 计算理论内存带宽
        float memoryBandwidth = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
        printf("  理论内存带宽:                  %.2f GB/s\n", memoryBandwidth);

        // 其他特性
        printf("\n  【其他特性】\n");
        printf("  支持并发 kernel 执行:          %s\n", prop.concurrentKernels ? "是" : "否");
        printf("  支持统一内存:                  %s\n", prop.unifiedAddressing ? "是" : "否");
        printf("  支持 ECC 内存:                 %s\n", prop.ECCEnabled ? "是" : "否");
    }

    printf("\n");
    printf("=");
    for (int i = 0; i < 50; i++) printf("=");
    printf("\n");
}
