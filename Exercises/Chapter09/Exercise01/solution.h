#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第九章：并行直方图 - 直方图实现
 * 
 * 包含5种实现：
 * 1. 顺序实现（CPU参考）
 * 2. 基础并行（图9.6）- 全局内存原子操作
 * 3. 私有化共享内存（图9.10）
 * 4. 线程粗化（图9.14）
 * 5. 线程粗化 + 内存合并访问
 */

// 直方图参数定义
// BIN_SIZE：每个桶包含的字母数（4个字母一组）
// NUM_BINS：桶的数量（26个字母 / 4 = 7个桶）
// COARSEN_FACTOR：线程粗化因子
#define BIN_SIZE 4
#define NUM_BINS ((26 + BIN_SIZE - 1) / BIN_SIZE)  // 7
#define COARSEN_FACTOR 32
#define BLOCK_SIZE 1024

/**
 * CPU 顺序实现（参考）
 */
void histogram_sequential(char* data, unsigned int length, unsigned int* histogram);

/**
 * 基础并行实现（图9.6）
 * 每个线程处理一个元素，直接对全局内存进行原子操作
 */
void histogram_parallel_basic(char* data, unsigned int length, unsigned int* histogram);

/**
 * 私有化共享内存实现（图9.10）
 * 每个 Block 使用共享内存维护私有直方图，最后合并到全局内存
 */
void histogram_parallel_private_shared(char* data, unsigned int length, unsigned int* histogram);

/**
 * 线程粗化实现（图9.14）
 * 每个线程处理多个连续元素
 */
void histogram_parallel_coarsening(char* data, unsigned int length, unsigned int* histogram);

/**
 * 线程粗化 + 内存合并访问
 * 使用 Grid-Stride Loop 实现更好的内存访问模式
 */
void histogram_parallel_coalesced(char* data, unsigned int length, unsigned int* histogram);

/**
 * 打印直方图内容（调试/展示用）
 */
void print_histogram(const unsigned int* histogram, const char* label);

#endif // SOLUTION_H
