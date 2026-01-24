/**
 * 第十一章：前缀和（扫描）- 测试程序
 * 
 * 参考：chapter-11/code/scan.cu
 * 
 * 测试所有扫描实现的正确性
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "solution.h"

const float TOLERANCE = 1e-3f;

// 验证两个数组是否接近
bool allclose(float* a, float* b, int N, float tolerance = TOLERANCE) {
    for (int i = 0; i < N; i++) {
        float allowed_error = tolerance + tolerance * fabs(b[i]);
        if (fabs(a[i] - b[i]) > allowed_error) {
            printf("  数组在索引 %d 处不匹配: %.2f vs %.2f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第十一章：前缀和（扫描）\n");
    printf("  Prefix Sum (Scan) Operations - Multiple Implementations\n");
    printf("  参考: chapter-11/code/scan.cu\n");
    printf("================================================================\n\n");

    // ==================== 小规模测试（单 Block）====================
    printf("=== 小规模测试（单 Block，999 元素）===\n\n");

    const unsigned int small_N = 999;
    float* X_small = new float[small_N];
    float* Y_seq = new float[small_N];
    float* Y_kogge = new float[small_N];
    float* Y_kogge_db = new float[small_N];
    float* Y_brent = new float[small_N];
    float* Y_three = new float[small_N];

    // 初始化：X[i] = i + 1
    for (unsigned int i = 0; i < small_N; i++) {
        X_small[i] = (float)(i + 1);
    }

    printf("1. CPU 顺序扫描...\n");
    scan_sequential(X_small, Y_seq, small_N);
    printf("   完成（前5个结果: %.0f, %.0f, %.0f, %.0f, %.0f）\n\n", 
           Y_seq[0], Y_seq[1], Y_seq[2], Y_seq[3], Y_seq[4]);

    printf("2. Kogge-Stone 扫描 (图11.3)...\n");
    scan_kogge_stone(X_small, Y_kogge, small_N);
    if (allclose(Y_kogge, Y_seq, small_N)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    printf("3. Kogge-Stone 双缓冲 (练习2)...\n");
    scan_kogge_stone_double_buffer(X_small, Y_kogge_db, small_N);
    if (allclose(Y_kogge_db, Y_seq, small_N)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    printf("4. Brent-Kung 扫描 (图11.4)...\n");
    scan_brent_kung(X_small, Y_brent, small_N);
    if (allclose(Y_brent, Y_seq, small_N)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    printf("5. 三阶段扫描...\n");
    scan_three_phase(X_small, Y_three, small_N);
    if (allclose(Y_three, Y_seq, small_N)) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    delete[] X_small;
    delete[] Y_seq;
    delete[] Y_kogge;
    delete[] Y_kogge_db;
    delete[] Y_brent;
    delete[] Y_three;

    // ==================== 大规模测试（多 Block）====================
    printf("=== 大规模测试（多 Block，100000 元素）===\n\n");

    const unsigned int large_N = 100000;
    float* X_large = new float[large_N];
    float* Y_seq_large = new float[large_N];
    float* Y_hier = new float[large_N];

    // 初始化：全 1
    for (unsigned int i = 0; i < large_N; i++) {
        X_large[i] = 1.0f;
    }

    printf("6. CPU 顺序扫描...\n");
    scan_sequential(X_large, Y_seq_large, large_N);
    printf("   完成（最后结果: %.0f）\n\n", Y_seq_large[large_N - 1]);

    printf("7. 分层扫描（任意长度）...\n");
    scan_hierarchical(X_large, Y_hier, large_N);
    if (allclose(Y_hier, Y_seq_large, large_N)) {
        printf("   ✅ 结果正确！（最后结果: %.0f）\n\n", Y_hier[large_N - 1]);
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    printf("8. Domino-Style 分层扫描（单kernel）...\n");
    float* Y_domino = new float[large_N];
    scan_hierarchical_domino(X_large, Y_domino, large_N);
    if (allclose(Y_domino, Y_seq_large, large_N)) {
        printf("   ✅ 结果正确！（最后结果: %.0f）\n\n", Y_domino[large_N - 1]);
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    delete[] X_large;
    delete[] Y_seq_large;
    delete[] Y_hier;
    delete[] Y_domino;

    printf("【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• Kogge-Stone：工作量 O(N log N)，步骤 O(log N)，适合低延迟\n");
    printf("• Brent-Kung：工作量 O(N)，更高效，分上扫和下扫两阶段\n");
    printf("• 双缓冲：消除写后读竞争，无需第二次 __syncthreads()\n");
    printf("• 三阶段：粗化策略，每线程处理多个元素\n");
    printf("• 分层扫描：支持任意长度，多 Block 协作\n");
    printf("• Domino-Style：单kernel实现，动态块索引分配，原子标志同步\n");
    printf("\n");

    printf("✅ 测试完成！\n\n");
    return 0;
}
