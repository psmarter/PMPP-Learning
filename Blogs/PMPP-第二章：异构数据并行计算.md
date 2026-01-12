---
title: PMPP-第二章：异构数据并行计算
date: 2026-01-11 19:38:29
tags:
  - CUDA
  - GPU编程
  - 并行计算
  - PMPP
  - 向量加法
categories:
  - PMPP
  - 知识分享
cover: /img/PMPP.jpg
---

## 前言

第二章"Heterogeneous Data Parallel Computing"（异构数据并行计算）是PMPP真正开始讲CUDA编程的地方。这章通过一个经典的向量加法例子，系统地介绍了CUDA C编程的核心要素：如何编写kernel函数、如何组织线程、如何管理设备内存。别看向量加法简单，它几乎涵盖了CUDA编程的所有基础概念。书中这章的内容非常扎实，代码示例也很完整，值得认真学习。

## 为什么从向量加法开始？

可能有人觉得向量加法太简单了，但它其实是理解数据并行计算的最佳入口。

### 什么是数据并行？

数据并行的核心思想很直白：对一组数据执行相同的操作，每个数据点的计算相互独立。向量加法就是个典型例子：

```
C[0] = A[0] + B[0]
C[1] = A[1] + B[1]
C[2] = A[2] + B[2]
...
C[n-1] = A[n-1] + B[n-1]
```

注意到没？每个元素的加法操作完全独立，C[0]的计算不需要等C[1]算完。这种独立性正是并行计算的黄金场景。

### 计算量与数据量

书中特别强调了一个概念：**算术强度**（Arithmetic Intensity）。对于向量加法来说：

- 每个元素需要：读取2个float（A[i]和B[i]）+ 写入1个float（C[i]）= 12字节数据传输
- 计算量：1次浮点加法

算术强度 = 1 FLOP / 12 Bytes ≈ 0.083 FLOP/Byte

这个数值相当低，属于典型的**内存受限**（Memory-Bound）问题。也就是说，GPU的计算单元会经常"饿着"等数据传输。虽然向量加法在GPU上能加速，但远达不到GPU的峰值性能。不过作为入门例子，它足够简单直观。

## 第一个CUDA程序：完整流程

### CPU版本：作为对比

先看传统的CPU实现，非常直白：

```c
void vecAddCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}
```

串行执行，一个接一个地计算。如果n=10000，就要循环10000次。

### CUDA版本：三步走

CUDA程序的典型结构可以总结为三个阶段：

#### 阶段1：Host端准备

```cuda
int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    // 1. 分配Host内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // 2. 初始化数据
    for (int i = 0; i < n; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }
```

这部分和普通C程序没啥区别。`h_`前缀是个好习惯，表示Host端的变量。

#### 阶段2：Device端准备

```cuda
    // 3. 分配Device内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // 4. 数据传输：Host → Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
```

这里引入了两个关键函数：

- **cudaMalloc**：在GPU显存上分配内存。注意参数是二级指针，因为需要修改指针本身的值。
- **cudaMemcpy**：在Host和Device之间拷贝数据。最后一个参数指定传输方向。

有个重要细节：`d_A`是Device内存的指针，但这个指针变量本身存储在Host内存中。也就是说，你不能在CPU代码中直接解引用`d_A[0]`，那会导致段错误。这是初学者常犯的错误。

#### 阶段3：执行计算并回传

```cuda
    // 5. 启动kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // 6. 结果传输：Device → Host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 7. 验证结果
    for (int i = 0; i < n; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            printf("Error at index %d\n", i);
        }
    }
    
    // 8. 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
```

这里的`<<<...>>>`语法是CUDA特有的，用于配置kernel启动参数。稍后详细讲。

## Kernel函数：GPU上的并行代码

### Kernel定义

```cuda
__global__ void vecAdd(float *A, float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```

看起来很简单，但每行都有讲究：

#### `__global__`修饰符

这个关键字告诉编译器：这是一个kernel函数，将在GPU上执行，由CPU调用。CUDA还有其他修饰符：

- `__device__`：GPU上执行，GPU上调用（类似GPU的库函数）
- `__host__`：CPU上执行，CPU上调用（就是普通函数，可省略）

#### 线程索引计算

```cuda
int i = blockDim.x * blockIdx.x + threadIdx.x;
```

这行代码是CUDA编程的精髓。要理解它，需要先理解CUDA的线程组织模型。

### CUDA线程层次结构

CUDA用两级层次组织线程：**Grid → Block → Thread**

#### 概念解释

- **Thread（线程）**：最小的执行单元，每个thread执行一次kernel函数
- **Block（线程块）**：一组threads的集合。同一个block中的threads可以共享数据、同步
- **Grid（网格）**：一组blocks的集合，代表整个kernel的执行

#### 一维组织示例

假设我们有10000个元素要处理，配置如下：

```cuda
int threadsPerBlock = 256;  // 每个block有256个threads
int blocksPerGrid = 40;     // 总共40个blocks
```

那么总共就有 40 × 256 = 10240 个threads。可视化一下：

```
Grid (40 blocks)
├── Block 0 (256 threads)
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
│   └── Thread 255
├── Block 1 (256 threads)
│   ├── Thread 0
│   └── ...
└── Block 39 (256 threads)
```

#### 全局索引的计算

每个thread需要知道自己应该处理数组的哪个元素。CUDA提供了内建变量：

- `blockIdx.x`：当前block在grid中的索引（0到39）
- `threadIdx.x`：当前thread在block中的索引（0到255）
- `blockDim.x`：block的大小（256）

所以全局索引的计算公式是：

```
i = blockIdx.x * blockDim.x + threadIdx.x
```

举例说明：

- Block 0的Thread 0：i = 0 × 256 + 0 = 0
- Block 0的Thread 1：i = 0 × 256 + 1 = 1
- Block 0的Thread 255：i = 0 × 256 + 255 = 255
- Block 1的Thread 0：i = 1 × 256 + 0 = 256
- Block 1的Thread 1：i = 1 × 256 + 1 = 257

看到了吗？每个thread得到一个唯一的索引，正好对应数组元素。

#### 边界检查的重要性

```cuda
if (i < n) {
    C[i] = A[i] + B[i];
}
```

这个检查至关重要。为什么？

因为我们通过`(n + threadsPerBlock - 1) / threadsPerBlock`计算block数量时，很可能总thread数多于数组元素数。例如n=10000，threadsPerBlock=256：

```
blocksPerGrid = (10000 + 255) / 256 = 40.07 → 40（向上取整）
总threads = 40 × 256 = 10240
```

多出来的240个threads（索引10000到10239）如果不检查边界，就会访问越界内存，导致错误或崩溃。

### 为什么选择256个threads？

你可能注意到我用的是256，而不是100或500。这不是随便选的，有几个考虑：

#### 1. Warp的倍数

GPU以32个threads为一组（称为warp）同时执行。如果block大小不是32的倍数，会浪费硬件资源。256 = 8 × 32，正好。

#### 2. 硬件限制

每个block的最大thread数通常是1024（取决于GPU架构）。256是个中等大小，既能充分利用硬件，又留有余地。

#### 3. 经验值

实践中发现，128到512之间的值通常性能较好。256算是个"安全"的选择。

不过具体项目中，最优值需要通过profiling来确定。后面章节会讲如何优化这些参数。

## 内存管理：Host与Device的鸿沟

### 两个独立的内存空间

这是异构编程最容易混淆的点：**Host和Device有各自独立的内存空间**。

- **Host Memory**：CPU的内存（DDR4/DDR5），由操作系统管理
- **Device Memory**：GPU的显存（GDDR6/HBM），由CUDA runtime管理

两者不能直接互访，必须通过显式的数据传输。

#### 错误示例

```cuda
float *d_A;
cudaMalloc((void**)&d_A, size);

// 错误！在CPU端访问GPU内存
d_A[0] = 1.0f;  // 段错误或未定义行为
```

这个错误我初学时犯过好几次。`d_A`是Device内存的指针，CPU不能直接解引用。

#### 正确做法

```cuda
float *h_A = (float*)malloc(size);
h_A[0] = 1.0f;  // CPU端修改

float *d_A;
cudaMalloc((void**)&d_A, size);
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);  // 传输到GPU
```

### cudaMemcpy的开销

数据传输不是免费的。PCIe总线的带宽虽然不低（PCIe 4.0 x16约32 GB/s），但相比GPU内存带宽（500+ GB/s）还是慢很多。

书中给了个数据：对于简单的向量加法，数据传输的时间可能是计算时间的数十倍。这就是为什么：

1. **尽量减少Host-Device传输次数**：一次传输大块数据，而不是多次传小块
2. **保持数据在GPU上**：如果后续操作也在GPU上进行，就别传回CPU了
3. **使用异步传输和计算重叠**：高级技巧，后面章节会讲

### Unified Memory：简化内存管理

从CUDA 6.0开始，NVIDIA引入了统一内存（Unified Memory）。使用`cudaMallocManaged`分配的内存可以在Host和Device之间自动迁移：

```cuda
float *data;
cudaMallocManaged(&data, size);

// CPU端访问
data[0] = 1.0f;

// GPU端访问（自动传输）
kernel<<<blocks, threads>>>(data);

// CPU端访问（自动传回）
printf("%f\n", data[0]);

cudaFree(data);
```

看起来很方便，但有几个注意点：

- **性能开销**：自动迁移有额外开销，显式管理可能更快
- **需要硬件支持**：较老的GPU不支持
- **页错误机制**：基于页面错误（page fault）实现，可能引入延迟

对于学习和原型开发，Unified Memory很友好。但生产环境中，还是建议显式管理内存以获得最佳性能。

## Kernel启动配置：`<<<grid, block>>>`

### 语法解析

```cuda
vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
```

这个三角括号语法是CUDA对C++的扩展。完整形式其实是：

```cuda
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);
```

- **gridDim**：Grid的维度，可以是1D、2D或3D
- **blockDim**：Block的维度，同样可以是1D、2D或3D
- **sharedMem**：每个block的共享内存大小（可选，默认0）
- **stream**：CUDA流，用于异步执行（可选，默认0）

向量加法只用了前两个参数，后面会遇到更复杂的用法。

### 多维组织

虽然向量加法用1D就够了，但很多应用需要多维组织。

#### 2D示例：矩阵加法

假设要处理1024×1024的矩阵：

```cuda
dim3 threadsPerBlock(16, 16);  // 每个block是16×16=256个threads
dim3 blocksPerGrid(64, 64);    // Grid是64×64=4096个blocks

matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
```

Kernel中的索引计算：

```cuda
__global__ void matrixAdd(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int i = row * WIDTH + col;  // 转换为1D索引
    
    if (row < HEIGHT && col < WIDTH) {
        C[i] = A[i] + B[i];
    }
}
```

注意`dim3`类型有三个成员：`.x`、`.y`、`.z`，默认值都是1。

#### 3D示例：体数据处理

医学成像、流体模拟等会用到3D组织：

```cuda
dim3 threadsPerBlock(8, 8, 8);  // 8×8×8 = 512 threads
dim3 blocksPerGrid(16, 16, 16); // 16×16×16 = 4096 blocks

volumeProcess<<<blocksPerGrid, threadsPerBlock>>>(d_volume);
```

索引计算：

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### 配置的限制

硬件有一些限制需要遵守：

- **每个block最多1024个threads**（计算能力3.0+）
- **Grid每个维度最多2^31-1个blocks**（x维度），其他维度是65535
- **每个block的共享内存有限**（通常48KB或96KB）

违反这些限制会导致kernel启动失败。可以用`cudaGetLastError()`检查错误。

## 错误处理：不能忽视的细节

CUDA的很多函数返回`cudaError_t`类型，表示操作是否成功。但它们不会自动抛出异常，需要显式检查。

### 包装宏

书中推荐定义一个检查宏：

```cuda
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
```

使用方法：

```cuda
CUDA_CHECK(cudaMalloc((void**)&d_A, size));
CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
```

这样一旦出错，会立即打印详细信息并退出。

### Kernel错误检查

Kernel启动本身不返回错误码，但可以在启动后检查：

```cuda
vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, n);
CUDA_CHECK(cudaGetLastError());  // 检查启动错误
CUDA_CHECK(cudaDeviceSynchronize());  // 同步并检查执行错误
```

`cudaDeviceSynchronize()`会等待之前的所有GPU操作完成，这对调试很有用。正式代码中可以去掉以提高性能（大部分CUDA操作是异步的）。

## 性能测量：究竟快了多少？

### 简单计时

CUDA提供了专门的计时函数：

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... 要测量的代码 ...
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("Execution time: %.3f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

这种方法的好处是能准确测量GPU上的执行时间，不受CPU-GPU异步的影响。

### 性能分析

书中给了个对比数据（基于NVIDIA Tesla V100）：

- **CPU（单线程）**：处理1000万元素向量，约20ms
- **GPU（CUDA）**：相同任务，约0.3ms

加速比约66倍！但注意，这**不包括数据传输时间**。如果算上`cudaMemcpy`：

- 总时间：约5ms（包含传输）
- 加速比：约4倍

这就是为什么说向量加法是内存受限的。真正的计算时间很短，大部分时间花在了数据传输上。

### 何时使用GPU？

从这个例子可以总结出一个经验法则：

GPU适合的场景：

- ✅ 大规模数据（百万级以上）
- ✅ 计算密集（每个数据有足够多的计算）
- ✅ 数据可以长时间留在GPU上
- ✅ 高度并行（无复杂依赖）

GPU不适合的场景：

- ❌ 小规模数据（几千个元素）
- ❌ 频繁的Host-Device传输
- ❌ 强串行依赖
- ❌ 复杂的控制流（大量分支）

## 实际应用的拓展

虽然向量加法看起来简单，但掌握它之后可以快速拓展到很多实际应用：

### 图像处理

每个像素的处理可以并行：

```cuda
__global__ void adjustBrightness(unsigned char *input, unsigned char *output, 
                                  int width, int height, float factor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int i = y * width + x;
        output[i] = min(255, (int)(input[i] * factor));
    }
}
```

调用时使用2D配置，每个thread处理一个像素。

### 机器学习

向量运算是神经网络的基础：

- **激活函数**：ReLU、Sigmoid等都可以逐元素并行计算
- **Batch Normalization**：每个样本独立处理
- **Element-wise操作**：加法、乘法、除法等

PyTorch和TensorFlow的底层就是这样实现的。例如`tensor1 + tensor2`在GPU上执行时，就类似我们写的向量加法kernel。

### 科学计算

- **数值积分**：每个区间独立计算
- **蒙特卡洛模拟**：每个样本独立采样
- **信号处理**：每个采样点独立变换（在FFT前）

## 调试技巧：几个常见坑

### 1. 忘记同步

```cuda
vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, n);
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);  // 这里会隐式同步
printf("Done\n");
```

`cudaMemcpy`会等待kernel完成，所以这里没问题。但如果换成：

```cuda
vecAdd<<<blocks, threads>>>(d_A, d_B, d_C, n);
// 没有同步！
// 这时kernel可能还没执行完
anotherKernel<<<blocks, threads>>>(d_C, d_D, n);
```

需要显式同步或使用stream控制执行顺序。

### 2. 索引计算错误

```cuda
// 错误：忘记乘以blockDim
int i = blockIdx.x + threadIdx.x;  // 很多threads会访问同一个元素！
```

这种错误不会立即崩溃，但结果会错误。调试时可以加打印：

```cuda
if (i < 10) {
    printf("Thread %d processing element %d\n", 
           threadIdx.x + blockIdx.x * blockDim.x, i);
}
```

### 3. 内存泄漏

```cuda
cudaMalloc((void**)&d_A, size);
// ... 使用 ...
// 忘记调用 cudaFree(d_A)！
```

GPU内存泄漏和CPU一样危险。用完记得释放。可以用`nvidia-smi`查看GPU内存占用。

### 4. 竞态条件

虽然向量加法没有这个问题（每个thread访问不同的内存位置），但这是并行编程的大坑：

```cuda
// 危险！多个threads可能同时写入
__global__ void bad(int *counter) {
    (*counter)++;  // 竞态条件！
}
```

正确做法是使用原子操作：

```cuda
__global__ void good(int *counter) {
    atomicAdd(counter, 1);  // 原子操作，线程安全
}
```

## 编译与运行

### 编译命令

使用nvcc（NVIDIA CUDA Compiler）：

```bash
nvcc -o vecAdd vecAdd.cu
```

常用选项：

- `-arch=sm_75`：指定计算能力（例如RTX 2080是7.5）
- `-O3`：开启优化
- `-g -G`：生成调试信息（CPU和GPU）

### 查看GPU信息

```bash
nvidia-smi
```

会显示GPU型号、驱动版本、显存使用情况等。

### 运行

```bash
./vecAdd
```

如果一切正常，会输出类似：

```
Vector addition completed
Execution time: 0.325 ms
Verification: PASSED
```

## 总结与思考

第二章虽然只讲了向量加法一个例子,但涵盖的知识点非常扎实:

**1. 掌握了CUDA编程的基本流程**
从内存分配、数据传输、kernel启动到结果回传,每一步都是必须掌握的基础。虽然流程看起来繁琐(相比直接调用库函数),但这种显式控制也给了我们优化的空间。

**2. 理解了线程组织模型**
Grid、Block、Thread的层次结构初看有点绕,但理解之后就会发现它的灵活性。索引计算`i = blockIdx.x * blockDim.x + threadIdx.x`必须烂熟于心,后面无数次会用到。

**3. 认识到内存管理的重要性**
数据传输的开销不能忽视。向量加法的例子清楚地展示了:简单的计算在GPU上很快,但如果算上传输时间,优势就没那么明显了。这也为后面章节的优化(减少传输、使用共享内存等)埋下伏笔。

**4. 并行思维的初步建立**
从"一个for循环遍历"转变成"n个线程同时执行",这是思维方式的转变。每个thread只关心自己的索引,不用管其他threads在干什么,这种"各自为政"的并行模式需要适应。

**几点个人体会:**

- **起点低但陡峭**:向量加法确实简单,但要把CUDA程序写对、运行起来,还是有不少细节要注意的。别着急,多写几遍自然就熟了。

- **性能不是唯一目标**:虽然GPU能加速计算,但不意味着所有代码都要往GPU上搬。要考虑开发成本、维护成本,以及实际的性能收益。

- **错误处理很必要**:初学时容易忽略错误检查,觉得"能跑就行"。但养成检查每个CUDA调用的习惯,能节省大量调试时间,尤其是遇到莫名其妙的错误时。

- **从简单到复杂**:书中选择向量加法作为第一个例子是有道理的。先把基础流程弄清楚,再逐步加入优化技巧(后续章节),学习曲线会平滑很多。

下一章应该会深入讲CUDA的内存层次和优化技术了。向量加法这个"内存受限"的问题,正好给优化留下了很大空间。期待学习如何使用共享内存、优化内存访问模式,把GPU的性能真正发挥出来。

---

**参考资料:**

- Hwu, W., Kirk, D., & El Hajj, I. (2022). *Programming Massively Parallel Processors: A Hands-on Approach* (4th Edition). Morgan Kaufmann.
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Runtime API Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/)
