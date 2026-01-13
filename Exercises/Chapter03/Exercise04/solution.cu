#include <stdio.h>
#include <cuda_runtime.h>
#include "../../../Common/utils.cuh"
#include "solution.h"

/**
 * CUDA Kernel: RGB 转灰度
 * 每个线程处理一个像素
 * 
 * 输入: RGB 图像，数据按行优先存储，交织格式 RGBRGBRGB...
 * 输出: 灰度图像，每个像素一个字节
 */
__global__ void rgbToGrayscaleKernel(unsigned char* Pin, unsigned char* Pout, 
                                     int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    const int CHANNELS = 3;  // RGB 三通道
    
    if (col < width && row < height) {
        // 行优先顺序
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;
        
        // 读取 RGB 值
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        
        // 转换为灰度值
        // 使用加权平均：Gray = 0.21*R + 0.71*G + 0.07*B
        Pout[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

/**
 * Host 函数：RGB 转灰度的完整流程
 */
void rgbToGrayscaleDevice(unsigned char* h_output, const unsigned char* h_input, 
                         int width, int height) {
    // 1. 分配设备内存
    unsigned char *d_input, *d_output;
    size_t input_bytes = width * height * 3 * sizeof(unsigned char);  // RGB 三通道
    size_t output_bytes = width * height * sizeof(unsigned char);     // 灰度单通道
    
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    
    // 2. 拷贝输入数据到设备
    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    
    // 3. 配置并启动 kernel
    // 使用 2D 线程块和网格来处理图像
    dim3 blockDim(32, 32);  // 32×32 = 1024 个线程per块
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    rgbToGrayscaleKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_LAST_CUDA_ERROR();
    
    // 4. 等待 GPU 完成
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 5. 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    
    // 6. 释放设备内存
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}
