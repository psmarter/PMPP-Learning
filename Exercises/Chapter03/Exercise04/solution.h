#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * RGB 转灰度的设备函数声明
 * 将彩色图像(RGB三通道)转换为灰度图像(单通道)
 * 
 * 灰度值计算公式: Gray = 0.21*R + 0.71*G + 0.07*B
 * 
 * @param h_output 输出灰度图像（主机内存）
 * @param h_input  输入 RGB 图像（主机内存）
 * @param width    图像宽度
 * @param height   图像高度
 */
void rgbToGrayscaleDevice(unsigned char* h_output, const unsigned char* h_input, 
                         int width, int height);

#endif // SOLUTION_H
