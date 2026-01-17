/**
 * 第九章：并行直方图 - 测试程序
 * 
 * 参考：chapter-09/code/histogram.cu
 * 
 * 测试所有5种实现的正确性
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "solution.h"

// Windows 兼容的 rand_r（如果不可用）
#ifdef _WIN32
int rand_r(unsigned int* seed) {
    *seed = *seed * 1103515245 + 12345;
    return (*seed / 65536) % 32768;
}
#endif

// 生成随机文本（小写字母）
char* generate_random_text(unsigned int length) {
    char* text = (char*)malloc((length + 1) * sizeof(char));
    if (text == NULL) {
        printf("内存分配失败！\n");
        return NULL;
    }

    unsigned int seed = 42;  // 固定种子保证可重复性
    for (unsigned int i = 0; i < length; i++) {
        text[i] = 'a' + (rand_r(&seed) % 26);
    }
    text[length] = '\0';
    return text;
}

// 验证两个直方图是否一致
bool verify_histogram(const unsigned int* A, const unsigned int* B, const char* nameA, const char* nameB) {
    bool match = true;
    for (int i = 0; i < NUM_BINS; i++) {
        if (A[i] != B[i]) {
            printf("  Bin %d 不匹配: %s=%u, %s=%u\n", i, nameA, A[i], nameB, B[i]);
            match = false;
        }
    }
    return match;
}

int main() {
    printf("\n");
    printf("================================================================\n");
    printf("  第九章：并行直方图\n");
    printf("  Histogram Computation - 5 Implementations\n");
    printf("  参考: chapter-09/code/histogram.cu\n");
    printf("================================================================\n\n");

    // 测试参数
    unsigned int length = 10000000;  // 1000万字符
    
    printf("配置:\n");
    printf("  数据长度: %u (%u M)\n", length, length / 1000000);
    printf("  BIN_SIZE: %d (每桶字母数)\n", BIN_SIZE);
    printf("  NUM_BINS: %d\n", NUM_BINS);
    printf("  BLOCK_SIZE: %d\n", BLOCK_SIZE);
    printf("  COARSEN_FACTOR: %d\n\n", COARSEN_FACTOR);

    // 分配内存
    unsigned int histo_seq[NUM_BINS] = {0};
    unsigned int histo_basic[NUM_BINS] = {0};
    unsigned int histo_shared[NUM_BINS] = {0};
    unsigned int histo_coarse[NUM_BINS] = {0};
    unsigned int histo_coalesced[NUM_BINS] = {0};

    // 生成随机数据
    printf("生成随机文本数据...\n");
    char* data = generate_random_text(length);
    if (data == NULL) {
        return 1;
    }
    printf("完成\n\n");

    printf("=== 正确性验证 ===\n\n");

    // 1. CPU 顺序实现（参考）
    printf("1. CPU 顺序实现...\n");
    histogram_sequential(data, length, histo_seq);
    printf("   完成\n\n");

    // 2. 基础并行
    printf("2. 基础并行 (图9.6)...\n");
    histogram_parallel_basic(data, length, histo_basic);
    if (verify_histogram(histo_seq, histo_basic, "CPU", "GPU")) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    // 3. 私有化共享内存
    printf("3. 私有化共享内存 (图9.10)...\n");
    histogram_parallel_private_shared(data, length, histo_shared);
    if (verify_histogram(histo_seq, histo_shared, "CPU", "GPU")) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    // 4. 线程粗化
    printf("4. 线程粗化 (图9.14)...\n");
    histogram_parallel_coarsening(data, length, histo_coarse);
    if (verify_histogram(histo_seq, histo_coarse, "CPU", "GPU")) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    // 5. 线程粗化 + 内存合并
    printf("5. 线程粗化 + 内存合并...\n");
    histogram_parallel_coalesced(data, length, histo_coalesced);
    if (verify_histogram(histo_seq, histo_coalesced, "CPU", "GPU")) {
        printf("   ✅ 结果正确！\n\n");
    } else {
        printf("   ❌ 结果不正确！\n\n");
    }

    // 显示直方图结果
    print_histogram(histo_seq, "直方图结果");

    printf("\n【关键概念】\n");
    printf("----------------------------------------------------------------\n");
    printf("• 原子操作：保证读-改-写的原子性，解决竞态条件\n");
    printf("• 私有化：每个 Block 用共享内存维护私有直方图\n");
    printf("• 共享内存原子：比全局内存原子操作快约 20 倍\n");
    printf("• 线程粗化：每个线程处理多个元素，减少 Block 数量\n");
    printf("• 内存合并：Grid-Stride Loop 使相邻线程访问相邻内存\n");
    printf("\n");

    // 释放内存
    free(data);

    printf("✅ 测试完成！\n\n");
    return 0;
}
