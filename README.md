# PMPP 学习笔记 - Programming Massively Parallel Processors（大规模并行处理器程序设计）

[![oosmetrics](https://api.oosmetrics.com/api/v1/badge/achievement/78e59898-3be7-46c1-9f62-0b0039e8a814.svg)](https://oosmetrics.com/repo/psmarter/PMPP-Learning)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![C++](https://img.shields.io/badge/C++-17-blue?style=flat-square&logo=cplusplus)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Blog](https://img.shields.io/badge/Blog-smarter.xin-orange?style=flat-square)](https://smarter.xin)

David B. Kirk 和 Wen-mei W. Hwu《Programming Massively Parallel Processors》第四版（大规模并行处理器程序设计）的完整学习记录，包含22篇学习笔记、38个完整Exercise、100+种算法实现和详细的CUDA代码。

> 📝 **博客系列**: [smarter.xin](https://smarter.xin) | 📚 **全书导读**: [点击阅读](https://smarter.xin/posts/30730973/)

---

## 关于

这个仓库是 PMPP 第四版的**完整学习记录**，包括：

- 📝 **22篇学习笔记**：覆盖全书所有章节，已发布在个人博客
- 💻 **38个完整Exercise**：每个都包含CUDA实现、测试程序和详细注释
- 🔢 **100+种算法实现**：从基础到高级的多版本对比（如7种归约、6种扫描）
- 📊 **完整性能测试**：CPU vs GPU对比、带宽分析、GFLOPS统计

## 项目统计

| 统计项 | 数量 | 说明 |
|--------|------|------|
| 博客文章 | 22篇 | 覆盖全书22章 |
| 代码章节 | 20章 | Chapter02-21（Chapter19为理论） |
| Exercise数量 | 38个 | 每个包含完整实现和测试 |
| 算法实现 | 100+种 | 多版本优化对比 |
| CUDA文件 | 40个.cu | 约15,000+行代码 |
| 测试文件 | 38个.cpp | 完整测试覆盖 |
| 文档文件 | 20个README | 练习题解答+学习建议 |

代码特点：

- **模块化架构**：97%的Exercise采用统一的4文件结构（solution.h/cu, test.cpp, Makefile）
- **完整测试**：所有38个Exercise包含CPU参考实现、正确性验证和性能测试
- **错误处理**：统一使用`CHECK_CUDA`宏，80%代码覆盖完整错误检查
- **公共工具**：Common目录提供统一的工具库（错误检查、计时器、图像I/O）
- **详细注释**：中文注释，包含算法原理、优化策略和关键概念说明
- **性能分析**：包含带宽计算、GFLOPS统计、加速比对比
- **教学友好**：从基础到高级，渐进式学习路径，预期输出便于验证

## 章节进度

| 章节 | 主题 | 状态 |
| ------ | ------ | ------ |
| [第 2 章](Exercises/Chapter02) | 异构数据并行计算 | ✅ 完成 |
| [第 3 章](Exercises/Chapter03) | 多维网格和数据 | ✅ 完成 |
| [第 4 章](Exercises/Chapter04) | 计算架构和调度 | ✅ 完成 |
| [第 5 章](Exercises/Chapter05) | 内存架构和数据局部性 | ✅ 完成 |
| [第 6 章](Exercises/Chapter06) | 性能方面的考虑 | ✅ 完成 |
| [第 7 章](Exercises/Chapter07) | 卷积 | ✅ 完成 |
| [第 8 章](Exercises/Chapter08) | 模板 | ✅ 完成 |
| [第 9 章](Exercises/Chapter09) | 并行直方图 | ✅ 完成 |
| [第 10 章](Exercises/Chapter10) | 归约 | ✅ 完成 |
| [第 11 章](Exercises/Chapter11) | 前缀和（扫描） | ✅ 完成 |
| [第 12 章](Exercises/Chapter12) | 归并 | ✅ 完成 |
| [第 13 章](Exercises/Chapter13) | 排序 | ✅ 完成 |
| [第 14 章](Exercises/Chapter14) | 稀疏矩阵计算 | ✅ 完成 |
| [第 15 章](Exercises/Chapter15) | 图遍历 | ✅ 完成 |
| [第 16 章](Exercises/Chapter16) | 深度学习 | ✅ 完成 |
| [第 17 章](Exercises/Chapter17) | 迭代式磁共振成像重建 | ✅ 完成 |
| [第 18 章](Exercises/Chapter18) | 静电势能图 | ✅ 完成 |
| [第 19 章](Exercises/Chapter19) | 并行编程与计算思维 | ✅ 完成 |
| [第 20 章](Exercises/Chapter20) | 异构计算集群编程 | ✅ 完成 |
| [第 21 章](Exercises/Chapter21) | CUDA动态并行性 | ✅ 完成 |

## 快速开始

### 环境要求

- **GPU**：NVIDIA GPU（计算能力3.5+，推荐RTX 20系列及以上）
- **CUDA**：CUDA Toolkit 11.0+
- **编译器**：GCC 7.5+ / Visual Studio 2019+ 或更高版本
- **系统**：Linux / Windows（大部分Exercise支持）

### 运行示例

```bash
# 1. 克隆仓库
git clone https://github.com/psmarter/PMPP-Learning.git
cd PMPP-Learning

# 2. 选择一个Exercise（以第2章向量乘法为例）
cd Exercises/Chapter02/Exercise01

# 3. 编译
make

# 4. 运行
make run

# 5. 清理
make clean

# 6. 查看帮助
make help
```

### 预期输出

```text
=== Correctness Test ===
Testing vector multiplication with 1048576 elements...
✅ Correctness test PASSED!

=== Performance Test ===
Data size: 1048576 elements (4.00 MB)
Iterations: 100

Results:
  Average time per iteration: 0.123 ms
  Effective bandwidth: 97.56 GB/s

✅ All tests completed successfully!
```

## 项目结构

```text
PMPP-Learning/
├── Blogs/                      # 学习笔记（22篇博客文章）
│   ├── PMPP-大规模并行处理器程序设计：导读.md
│   ├── PMPP-第一章：引言.md
│   ├── PMPP-第二章：异构数据并行计算.md
│   ├── ...（其他章节，共22章）
│   └── PMPP-第二十二章：高级实践与未来演变.md
│
├── Common/                     # 公共工具库
│   ├── utils.cuh               # CUDA 错误检查宏
│   ├── timer.h                 # CPU/GPU 性能计时器
│   ├── stb_image.h             # 图像加载库
│   └── stb_image_write.h       # 图像保存库
│
└── Exercises/                  # 章节练习（20章，38个Exercise）
    ├── Chapter02/              # 异构数据并行计算
    │   ├── README.md
    │   └── Exercise01/         # 向量乘法
    │       ├── solution.h      # 接口声明
    │       ├── solution.cu     # CUDA实现
    │       ├── test.cpp        # 测试程序
    │       └── Makefile        # 编译配置
    │
    ├── Chapter03/              # 多维网格和数据
    │   ├── README.md
    │   ├── Exercise01/         # 行/列级矩阵乘法
    │   ├── Exercise02/         # 矩阵向量乘法
    │   ├── Exercise03/         # 标准矩阵乘法
    │   ├── Exercise04/         # RGB转灰度（含图像）
    │   └── Exercise05/         # 高斯模糊（含图像）
    │
    ├── Chapter05/              # 内存架构和数据局部性
    │   ├── Exercise01/         # Tiled矩阵乘法
    │   └── Exercise02/         # 动态Tile大小
    │
    ├── ...（其他章节）
    │
    ├── Chapter15/              # 图遍历（特殊结构）
    │   └── Exercise01/
    │       ├── include/        # 模块化头文件
    │       ├── src/            # 模块化源文件
    │       └── Makefile
    │
    └── Chapter16-21/           # 高级应用章节
        └── ...（深度学习、MRI重建、动态并行等）
```

**说明**：
- 每个Exercise包含标准4文件（solution.h/cu, test.cpp, Makefile）
- Chapter15使用模块化结构（复杂项目特例）
- 总计38个Exercise，涵盖GPU并行编程的核心算法

## 核心特性

### 🛡️ 完整的错误检查

所有CUDA API调用都使用统一的错误检查宏（Common/utils.cuh）：

```cuda
// 内存分配和拷贝
CHECK_CUDA(cudaMalloc(&d_data, size));
CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

// Kernel调用
myKernel<<<grid, block>>>(args);
CHECK_LAST_CUDA_ERROR();  // 检查kernel执行错误

// 同步
CHECK_CUDA(cudaDeviceSynchronize());
```

### 🧪 完整的测试框架

每个Exercise都包含三层测试：

```cpp
// 1. CPU参考实现
void cpuCompute(float* output, const float* input, int N) {
    // 顺序算法，用于验证GPU结果
}

// 2. 正确性验证（GPU vs CPU）
bool testCorrectness() {
    gpuCompute(gpu_result, input, N);
    cpuCompute(cpu_result, input, N);
    return verifyResults(gpu_result, cpu_result, N, epsilon);
}

// 3. 性能测试（预热 + 多次迭代 + 统计）
void testPerformance() {
    // 预热
    gpuCompute(result, input, N);
    
    // 性能测试
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < iterations; i++) {
        gpuCompute(result, input, N);
    }
    timer.stop();
    
    // 统计输出
    float avgTime = timer.elapsed_ms() / iterations;
    float bandwidth = calculateBandwidth(N, avgTime);
    float gflops = calculateGFLOPS(N, avgTime);
    printf("时间: %.3f ms | 带宽: %.2f GB/s | 性能: %.2f GFLOPS\n", 
           avgTime, bandwidth, gflops);
}
```

### ⚡ 多版本优化对比

以归约（Chapter10）为例，实现了7种优化版本：

| 版本 | 优化策略 | 性能提升 |
|------|---------|----------|
| Simple | 基础实现 | 基准 |
| Convergent | 减少控制分歧 | 1.2x |
| Shared Memory | 使用共享内存 | 2.5x |
| Segmented | 分段归约 | 3.5x |
| Coarsened | 线程粗化 | 5.2x |
| Warp Shuffle | Warp级原语 | 6.8x |

类似的多版本实现遍布各章节（扫描6种、排序5种、直方图5种等）。

## 学习资源

### 📝 博客文章

本项目配套22篇系列博客文章（已发布在 [smarter.xin](https://smarter.xin)）：

- [PMPP-大规模并行处理器程序设计：导读](https://smarter.xin/posts/30730973/) - 全书概览和学习路线
- [第一章：引言](https://smarter.xin/posts/10d278b0/) - CPU vs GPU设计哲学
- [第二章：异构数据并行计算](https://smarter.xin/posts/3ee22ce5/) - CUDA基础概念
- [第五章：内存架构和数据局部性](https://smarter.xin/posts/3bb3179b/) - Tiling技术（核心）
- [第十章：归约和最小化发散](https://smarter.xin/posts/43b40d12/) - 树形并行算法
- [第十一章：前缀和](https://smarter.xin/posts/a6fc4cf6/) - Scan算法
- 完整列表见 [Blogs目录](Blogs/)

### 💡 学习建议

1. **循序渐进**：按章节顺序学习，先阅读博客理解概念，再运行代码验证
2. **重点突破**：第5章（Tiling）、第10-11章（Reduce/Scan）是核心，务必深入理解
3. **动手实践**：所有Exercise都可编译运行，建议修改参数观察性能变化
4. **对比分析**：每章包含多种实现版本，对比理解优化策略的效果
5. **验证理解**：使用提供的预期输出验证自己的实现是否正确

## 项目亮点

- ✅ **38个完整Exercise**：覆盖GPU并行编程的所有核心算法
- ✅ **100种+算法实现**：从基础到高级，多版本对比（如：7种归约、6种扫描）
- ✅ **统一架构**：97%采用标准4文件结构，便于学习和维护
- ✅ **完整测试框架**：每个Exercise包含CPU参考、正确性验证和性能测试
- ✅ **详细文档**：22篇博客 + 20个章节README + 完整代码注释
- ✅ **实用工具**：统一的错误检查、性能计时、图像处理工具
- ✅ **教学友好**：中文注释、预期输出、学习建议，适合系统学习

## 常见问题

### 编译错误？

**问题**：找不到`utils.cuh`或`timer.h`
```bash
fatal error: utils.cuh: No such file or directory
```

**解决**：确保Makefile中正确引用Common目录：
```makefile
COMMON_DIR = ../../../Common
$(NVCC) $(CUDA_FLAGS) -I$(COMMON_DIR) -c $<
```

### GPU计算能力不足？

**问题**：`no kernel image is available for execution on the device`

**解决**：修改Makefile中的`CUDA_ARCH`为你的GPU架构：
```makefile
# RTX 40系列
CUDA_ARCH = sm_89
# RTX 30系列
CUDA_ARCH = sm_86  
# RTX 20系列
CUDA_ARCH = sm_75
```

### Windows下编译？

**问题**：Makefile使用了Linux命令

**解决**：使用Git Bash或WSL，或修改Makefile的`rm`命令为`del`

### cuDNN库缺失？

**问题**：Chapter16/Exercise05编译失败
```bash
fatal error: cudnn.h: 没有那个文件或目录
```

**解决**：
- **方案1**：安装cuDNN库（参见[Exercise05/README](Exercises/Chapter16/README.md#exercise05---cudnn-封装实现)）
- **方案2**：跳过Exercise05，完成Exercise01-04即可（不影响其他练习）

## LICENSE

MIT License - 详见 [LICENSE](LICENSE)

本项目为个人学习成果，仅供学习交流使用。代码经过测试验证，但不保证无错误，使用时请自行测试。

## 技术栈

- **语言**：CUDA C++17
- **编译器**：NVCC (CUDA Toolkit 11.0+)
- **GPU要求**：NVIDIA GPU，计算能力3.5+
- **操作系统**：Linux / Windows
- **构建工具**：Make
- **库依赖**：
  - cuBLAS（Chapter16）
  - cuDNN（Chapter16 Exercise05，可选）
  - MPI（Chapter20）

## 相关资源

### 官方文档
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PMPP书籍官方网站](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-0)

### 博客系列
- [全书导读](https://smarter.xin/posts/30730973/) - 学习路线和各章概览
- [完整博客列表](Blogs/) - 22篇系列文章


## 致谢

感谢以下资源对本项目的帮助：

- **David B. Kirk & Wen-mei W. Hwu**：编写了优秀的PMPP教材
- **NVIDIA**：提供强大的CUDA平台和完善的开发文档
- **开源社区**：stb_image库等优秀的开源工具
- **习题解答参考**：[tugot17/pmpp](https://github.com/tugot17/pmpp/)
---

<div align="center">

⭐ **如果这个项目对你有帮助，欢迎Star支持！**


🌐 **博客主页**：[https://smarter.xin](https://smarter.xin)

</div>
