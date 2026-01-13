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
 * CPU 版本的高斯模糊（均值滤波）
 */
void gaussianBlurCPU(unsigned char* output, const unsigned char* input,
                    int width, int height, int blur_size) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int pixelValues = 0;
            int pixels = 0;
            
            for (int blurRow = -blur_size; blurRow <= blur_size; ++blurRow) {
                for (int blurCol = -blur_size; blurCol <= blur_size; ++blurCol) {
                    int currCol = col + blurCol;
                    int currRow = row + blurRow;
                    
                    if (currCol >= 0 && currCol < width && currRow >= 0 && currRow < height) {
                        pixelValues += input[currRow * width + currCol];
                        ++pixels;
                    }
                }
            }
            
            output[row * width + col] = (unsigned char)(pixelValues / pixels);
        }
    }
}

/**
 * 验证结果
 */
bool verifyResults(const unsigned char* gpu_result, const unsigned char* cpu_result, int size) {
    int max_diff = 0;
    int diff_count = 0;
    for (int i = 0; i < size; ++i) {
        int diff = abs((int)gpu_result[i] - (int)cpu_result[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0) {
            diff_count++;
            if (diff_count < 10) {
                printf("❌ Mismatch at pixel %d: GPU = %d, CPU = %d\n", 
                       i, gpu_result[i], cpu_result[i]);
            }
        }
    }
    
    if (diff_count > 0) {
        printf("共发现 %d 个不匹配像素\n", diff_count);
        return false;
    }
    
    printf("最大差异: %d\n", max_diff);
    return true;
}

/**
 * RGB 转灰度（用于图像预处理）
 */
void rgbToGrayscale(unsigned char* output, const unsigned char* input, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = input[i * 3];
        unsigned char g = input[i * 3 + 1];
        unsigned char b = input[i * 3 + 2];
        output[i] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

/**
 * 正确性测试
 */
bool testCorrectness(const char* image_path, int blur_size) {
    printf("\n=== 正确性测试 ===\n");
    printf("加载图像: %s\n", image_path);
    
    // 加载彩色图像
    int width, height, channels;
    unsigned char* h_rgb = stbi_load(image_path, &width, &height, &channels, 3);
    
    if (!h_rgb) {
        printf("❌ 无法加载图像: %s\n", image_path);
        printf("   请确保 Grace_Hopper.jpg 文件在当前目录\n");
        return false;
    }
    
    printf("图像大小: %d×%d\n", width, height);
    printf("模糊窗口: %d×%d\n\n", 2*blur_size+1, 2*blur_size+1);
    
    // 转换为灰度图
    unsigned char* h_gray = (unsigned char*)malloc(width * height);
    rgbToGrayscale(h_gray, h_rgb, width, height);
    stbi_image_free(h_rgb);
    
    // 分配输出内存
    unsigned char* h_output_gpu = (unsigned char*)malloc(width * height);
    unsigned char* h_output_cpu = (unsigned char*)malloc(width * height);
    
    // GPU 计算
    printf("执行 GPU 计算...\n");
    gaussianBlurDevice(h_output_gpu, h_gray, width, height, blur_size);
    
    // CPU 验证
    printf("执行 CPU 验证...\n");
    gaussianBlurCPU(h_output_cpu, h_gray, width, height, blur_size);
    
    // 保存结果
    printf("保存结果图像...\n");
    stbi_write_jpg("output_blurred_gpu.jpg", width, height, 1, h_output_gpu, 95);
    stbi_write_jpg("output_blurred_cpu.jpg", width, height, 1, h_output_cpu, 95);
    stbi_write_jpg("output_original_gray.jpg", width, height, 1, h_gray, 95);
    printf("已保存: output_blurred_gpu.jpg, output_blurred_cpu.jpg, output_original_gray.jpg\n\n");
    
    // 验证
    bool correct = verifyResults(h_output_gpu, h_output_cpu, width * height);
    printf("%s\n", correct ? "✅ 正确性测试通过" : "❌ 正确性测试失败");
    
    // 释放内存
    free(h_gray);
    free(h_output_gpu);
    free(h_output_cpu);
    
    return correct;
}

/**
 * 性能测试
 */
void testPerformance(const char* image_path, int blur_size) {
    printf("\n=== 性能测试 ===\n");
    
    // 加载并转换为灰度图
    int width, height, channels;
    unsigned char* h_rgb = stbi_load(image_path, &width, &height, &channels, 3);
    
    if (!h_rgb) {
        printf("❌ 无法加载图像进行性能测试\n");
        return;
    }
    
    unsigned char* h_gray = (unsigned char*)malloc(width * height);
    rgbToGrayscale(h_gray, h_rgb, width, height);
    stbi_image_free(h_rgb);
    
    int iterations = 50;
    printf("图像大小: %d×%d\n", width, height);
    printf("模糊窗口: %d×%d\n", 2*blur_size+1, 2*blur_size+1);
    printf("迭代次数: %d\n\n", iterations);
    
    // 分配输出内存
    unsigned char* h_output = (unsigned char*)malloc(width * height);
    
    // 预热
    gaussianBlurDevice(h_output, h_gray, width, height, blur_size);
    
    // 性能测试
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; ++i) {
        gaussianBlurDevice(h_output, h_gray, width, height, blur_size);
    }
    timer.stop();
    
    double avg_time = timer.elapsed_ms() / iterations;
    double operations_per_pixel = (2*blur_size+1) * (2*blur_size+1);  // 每个像素的操作数
    double total_ops = width * height * operations_per_pixel;
    double gops = (total_ops / avg_time) / 1e6;  // G操作/秒
    
    printf("结果:\n");
    printf("  平均时间: %.3f ms\n", avg_time);
    printf("  吞吐量: %.2f MPixels/s\n", (width * height / avg_time) / 1e3);
    printf("  计算性能: %.2f GOps/s\n", gops);
    
    // 释放内存
    free(h_gray);
    free(h_output);
}

int main() {
    printf("==================================================\n");
    printf("第三章 - 练习5: 高斯模糊\n");
    printf("==================================================\n");
    
    printDeviceInfo();
    
    const char* image_path = "Grace_Hopper.jpg";
    int blur_size = 5;  // 11×11 窗口
    
    if (!testCorrectness(image_path, blur_size)) {
        printf("\n❌ 正确性测试失败！\n");
        return 1;
    }
    
    testPerformance(image_path, blur_size);
    
    printf("\n✅ 所有测试完成！\n");
    printf("\n提示: 此实现使用简化的均值滤波。\n");
    printf("      输出图像已保存，可对比查看效果。\n\n");
    
    return 0;
}
