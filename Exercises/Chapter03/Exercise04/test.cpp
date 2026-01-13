#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../../Common/utils.cuh"
#include "../../../Common/timer.h"
#include "solution.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../../Common/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../Common/stb_image_write.h"

/**
 * CPU 版本的 RGB 转灰度
 */
void rgbToGrayscaleCPU(unsigned char* output, const unsigned char* input, 
                       int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = input[i * 3];
        unsigned char g = input[i * 3 + 1];
        unsigned char b = input[i * 3 + 2];
        output[i] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

/**
 * 验证结果
 */
bool verifyResults(const unsigned char* gpu_result, const unsigned char* cpu_result, 
                  int size) {
    int max_diff = 0;
    int diff_count = 0;
    for (int i = 0; i < size; ++i) {
        int diff = abs((int)gpu_result[i] - (int)cpu_result[i]);
        if (diff > max_diff) max_diff = diff;
        // 允许 ±1 的误差（由于浮点计算）
        if (diff > 1) {
            diff_count++;
            if (diff_count < 10) {  // 只打印前10个错误
                printf("❌ Mismatch at pixel %d: GPU = %d, CPU = %d\n", 
                       i, gpu_result[i], cpu_result[i]);
            }
        }
    }
    
    if (diff_count > 0) {
        printf("共发现 %d 个不匹配像素\n", diff_count);
        return false;
    }
    
    printf("最大差异: %d (允许±1)\n", max_diff);
    return true;
}

/**
 * 正确性测试
 */
bool testCorrectness(const char* image_path) {
    printf("\n=== 正确性测试 ===\n");
    printf("加载图像: %s\n", image_path);
    
    // 加载图像
    int width, height, channels;
    unsigned char* h_input = stbi_load(image_path, &width, &height, &channels, 3);
    
    if (!h_input) {
        printf("❌ 无法加载图像: %s\n", image_path);
        printf("   请确保 Grace_Hopper.jpg 文件在当前目录\n");
        return false;
    }
    
    printf("图像大小: %d×%d, 通道数: %d\n\n", width, height, channels);
    
    // 分配输出内存
    unsigned char* h_output_gpu = (unsigned char*)malloc(width * height);
    unsigned char* h_output_cpu = (unsigned char*)malloc(width * height);
    
    // GPU 计算
    printf("执行 GPU 计算...\n");
    rgbToGrayscaleDevice(h_output_gpu, h_input, width, height);
    
    // CPU 验证
    printf("执行 CPU 验证...\n");
    rgbToGrayscaleCPU(h_output_cpu, h_input, width, height);
    
    // 保存结果
    printf("保存结果图像...\n");
    stbi_write_jpg("output_gpu.jpg", width, height, 1, h_output_gpu, 95);
    stbi_write_jpg("output_cpu.jpg", width, height, 1, h_output_cpu, 95);
    printf("已保存: output_gpu.jpg, output_cpu.jpg\n\n");
    
    // 验证
    bool correct = verifyResults(h_output_gpu, h_output_cpu, width * height);
    printf("%s\n", correct ? "✅ 正确性测试通过" : "❌ 正确性测试失败");
    
    // 释放内存
    stbi_image_free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    
    return correct;
}

/**
 * 性能测试
 */
void testPerformance(const char* image_path) {
    printf("\n=== 性能测试 ===\n");
    
    // 加载图像
    int width, height, channels;
    unsigned char* h_input = stbi_load(image_path, &width, &height, &channels, 3);
    
    if (!h_input) {
        printf("❌ 无法加载图像进行性能测试\n");
        return;
    }
    
    int iterations = 100;
    printf("图像大小: %d×%d\n", width, height);
    printf("迭代次数: %d\n\n", iterations);
    
    // 分配输出内存
    unsigned char* h_output = (unsigned char*)malloc(width * height);
    
    // 预热
    rgbToGrayscaleDevice(h_output, h_input, width, height);
    
    // 性能测试
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        rgbToGrayscaleDevice(h_output, h_input, width, height);
    }
    timer.stop();
    
    double avg_time = timer.elapsed_ms() / iterations;
    size_t bytes_read = width * height * 3;  // RGB 输入
    size_t bytes_written = width * height;   // 灰度输出
    double bandwidth = ((bytes_read + bytes_written) / avg_time) / 1e6;  // GB/s
    
    printf("结果:\n");
    printf("  平均时间: %.3f ms\n", avg_time);
    printf("  带宽: %.2f GB/s\n", bandwidth);
    printf("  吞吐量: %.2f MPixels/s\n", (width * height / avg_time) / 1e3);
    
    // 释放内存
    stbi_image_free(h_input);
    free(h_output);
}

int main() {
    printf("==================================================\n");
    printf("第三章 - 练习4: RGB 转灰度\n");
    printf("==================================================\n");
    
    printDeviceInfo();
    
    const char* image_path = "Grace_Hopper.jpg";
    
    if (!testCorrectness(image_path)) {
        printf("\n❌ 正确性测试失败！\n");
        return 1;
    }
    
    testPerformance(image_path);
    
    printf("\n✅ 所有测试完成！\n");
    printf("\n提示: 输出图像已保存为 output_gpu.jpg 和 output_cpu.jpg\n\n");
    
    return 0;
}
