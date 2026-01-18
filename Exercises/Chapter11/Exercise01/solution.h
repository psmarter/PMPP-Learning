#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 第十一章：前缀和（扫描）- 并行扫描实现
 * 
 * 包含多种扫描实现：
 * 1. 顺序实现（CPU参考）
 * 2. Kogge-Stone 扫描（图11.3）
 * 3. Kogge-Stone 双缓冲扫描（练习2）
 * 4. Brent-Kung 扫描（图11.4）
 * 5. 三阶段扫描
 * 6. 分层扫描（支持任意长度）
 */

// Block 尺寸和粗化因子
#define SECTION_SIZE 1024
#define COARSE_FACTOR 4

// ====================== 基础扫描（单 Block）======================

/**
 * CPU 顺序包含扫描（参考）
 */
void scan_sequential(float* X, float* Y, unsigned int N);

/**
 * Kogge-Stone 扫描（图11.3）
 * 限制：仅支持 <= 1024 元素
 * 特点：工作量 O(N log N)，步骤 O(log N)
 */
void scan_kogge_stone(float* X, float* Y, unsigned int N);

/**
 * Kogge-Stone 双缓冲扫描（练习2）
 * 使用双缓冲避免写后读竞争
 */
void scan_kogge_stone_double_buffer(float* X, float* Y, unsigned int N);

/**
 * Brent-Kung 扫描（图11.4）
 * 特点：工作量 O(N)，更高效
 */
void scan_brent_kung(float* X, float* Y, unsigned int N);

/**
 * 三阶段扫描（粗化）
 * 每线程处理多个元素
 */
void scan_three_phase(float* X, float* Y, unsigned int N);

// ====================== 分层扫描（任意长度）======================

/**
 * 分层 Kogge-Stone 扫描
 * 支持任意长度，使用多 Block
 */
void scan_hierarchical(float* X, float* Y, unsigned int N);

#endif // SOLUTION_H
