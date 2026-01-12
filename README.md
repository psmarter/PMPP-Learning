# PMPP 学习笔记 - Programming Massively Parallel Processors

[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![C++](https://img.shields.io/badge/C++-17-blue?style=flat-square&logo=cplusplus)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Blog](https://img.shields.io/badge/Blog-smarter.xin-orange?style=flat-square)](https://smarter.xin)

David Kirk 和 Wen-mei Hwu《Programming Massively Parallel Processors》第四版的学习记录，包含练习题解答、CUDA代码实现和学习笔记。

> 📝 **博客主页**: [https://smarter.xin](https://smarter.xin)

---

## 关于

这个仓库记录了学习 PMPP 第四版的过程，包括：

- 📝 每章学习笔记和核心概念总结
- 💻 练习题的详细解答和推导过程
- ⚙️ CUDA 代码实现（包含完整错误检查和性能测试）
- 📊 性能测试和分析结果

代码特点：

- 模块化组织（kernel 实现和测试分离）
- 使用共享头文件，避免声明重复
- 完整的错误检查机制
- 详细的中文注释
- 正确性验证 + 性能测试
- 符合现代 C++ 最佳实践
- 支持 Linux 和 Windows

## 章节进度

| 章节 | 主题 | 状态 |
| ------ | ------ | ------ |
| [第 2 章](Exercises/Chapter02) | 异构数据并行计算 | ✅ 完成 |
| 第 3 章 | 多维网格和数据 | 🚧 进行中 |
| 第 4 章 | 计算架构和调度 | 📅 计划中 |

## 快速开始

环境要求：

- NVIDIA GPU (计算能力 3.5+)
- CUDA Toolkit 11.0+
- GCC 7.5+ / Visual Studio 2019+

运行示例：

```bash
# 克隆仓库
git clone https://github.com/psmarter/PMPP-Learning.git
cd PMPP-Learning

# 进入练习目录
cd Exercises/Chapter02/Exercise01

# 编译运行
make
make run
```

## 项目结构

```text
PMPP-Learning/
├── Blogs/                      # 学习笔记
│   └── PMPP-第二章：异构数据并行计算.md
├── Common/                     # 公共工具
│   ├── utils.cuh               # CUDA 错误检查
│   └── timer.h                 # 性能计时器（支持 CPU 和 CUDA 计时）
└── Exercises/                  # 章节练习
    └── Chapter02/
        ├── README.md           # 练习题解答
        └── Exercise01/         # 向量乘法练习
            ├── solution.h      # 函数声明头文件
            ├── solution.cu     # CUDA 实现
            ├── test.cpp        # 测试代码
            └── Makefile        # 编译脚本
```

## 代码示例

### 错误检查

所有 CUDA API 调用都包含错误检查：

```cuda
CHECK_CUDA(cudaMalloc(&d_data, size));
CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

myKernel<<<grid, block>>>(args);
CHECK_LAST_CUDA_ERROR();
```

### 性能测试

每个练习都包含正确性验证和性能测试：

```cpp
// 正确性: GPU vs CPU 结果对比
bool testCorrectness() {
    gpuCompute(gpu_result, input, N);
    cpuCompute(cpu_result, input, N);
    return verifyResults(gpu_result, cpu_result, N);
}

// 性能: 多次迭代取平均 + 带宽计算
void testPerformance() {
    CudaTimer timer;
    timer.start();
    for (int i = 0; i < 100; i++) {
        gpuCompute(result, input, N);
    }
    timer.stop();
    printf("时间: %.3f ms\n", timer.elapsed_ms() / 100);
    printf("带宽: %.2f GB/s\n", calculateBandwidth());
}
```

## 学习建议

1. 按章节顺序学习，先看博客笔记理解概念
2. 独立思考练习题后再看解答
3. 运行代码观察实际效果
4. 尝试修改参数（块大小、数据量）进行实验
5. 对比 CPU 和 GPU 的性能差异

## LICENSE

MIT License - 详见 [LICENSE](LICENSE)

代码实现为个人学习成果，仅供学习交流使用。

## 相关资源

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [书籍官方网站](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-0)

## 致谢

练习题解答参考了 [tugot17/pmpp](https://github.com/tugot17/pmpp) 仓库。
