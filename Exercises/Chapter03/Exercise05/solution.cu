#include <stdio.h>
#include <cuda_runtime.h>
#include "../../../Common/utils.cuh"
#include "solution.h"

/**
 * CUDA Kernel: 高斯模糊（简化版本，使用均值滤波）
 * 每个线程处理一个像素
 * 
 * 对每个像素，计算其周围 (2*blur_size+1)×(2*blur_size+1) 窗口内所有像素的平均值
 */
__global__ void blurKernel(unsigned char* Pin, unsigned char* Pout,
                          int width, int height, int blur_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int pixelValues = 0;
        int pixels = 0;
        
        // 遍历模糊窗口
        for (int blurRow = -blur_size; blurRow <= blur_size; ++blurRow) {
            for (int blurCol = -blur_size; blurCol <= blur_size; ++blurCol) {
                int currCol = col + blurCol;
                int currRow = row + blurRow;
                
                // 边界检查
                if (currCol >= 0 && currCol < width && currRow >= 0 && currRow < height) {
                    pixelValues += Pin[currRow * width + currCol];
                    ++pixels;
                }
            }
        }
        
        // 计算平均值
        Pout[row * width + col] = (unsigned char)(pixelValues / pixels);
    }
}

/**
 * Host 函数：高斯模糊的完整流程
 */
void gaussianBlurDevice(unsigned char* h_output, const unsigned char* h_input,
                       int width, int height, int blur_size) {
    // 1. 分配设备内存
    unsigned char *d_input, *d_output;
    size_t bytes = width * height * sizeof(unsigned char);
    
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    
    // 2. 拷贝输入数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // 3. 配置并启动 kernel
    dim3 blockDim(16, 16);  // 16×16 = 256 个线程per块
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    blurKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, blur_size);
    CHECK_LAST_CUDA_ERROR();
    
    // 4. 等待 GPU 完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    
    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}
