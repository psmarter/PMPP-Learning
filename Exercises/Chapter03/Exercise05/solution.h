#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 高斯模糊的设备函数声明
 * 对灰度图像进行高斯模糊处理
 * 
 * @param h_output   输出模糊图像（主机内存）
 * @param h_input    输入图像（主机内存）
 * @param width      图像宽度
 * @param height     图像高度
 * @param blur_size  模糊半径（模糊窗口为 (2*blur_size+1)×(2*blur_size+1)）
 */
void gaussianBlurDevice(unsigned char* h_output, const unsigned char* h_input,
                       int width, int height, int blur_size);

#endif // SOLUTION_H
