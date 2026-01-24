#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "solution.h"

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    // 默认参数
    int dimx = 48, dimy = 48, dimz = 40, nreps = 10;
    int pid = -1, np = -1;
    
    // 初始化 MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    // 调试输出
    printf("[DEBUG] Process %d: MPI_Comm_size = %d\n", pid, np);
    fflush(stdout);
    
    // 检查进程数（至少需要2个：1计算节点+1数据服务器）
    if (np < 2) {
        if (pid == 0) {
            printf("================================================================\n");
            printf("  第二十章：异构计算集群编程\n");
            printf("  MPI + CUDA Distributed Stencil Computation\n");
            printf("================================================================\n\n");
            printf("错误：需要至少 2 个 MPI 进程\n");
            printf("用法：mpirun -np 3 ./stencil_mpi\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // 打印信息（仅数据服务器）
    if (pid == np - 1) {
        printf("================================================================\n");
        printf("  第二十章：异构计算集群编程\n");
        printf("  MPI + CUDA Distributed Stencil Computation\n");
        printf("================================================================\n\n");
        printf("配置信息:\n");
        printf("  网格尺寸: %d x %d x %d\n", dimx, dimy, dimz);
        printf("  迭代次数: %d\n", nreps);
        printf("  MPI 进程数: %d (%d 计算节点 + 1 数据服务器)\n", np, np - 1);
        printf("  Halo 大小: %d\n\n", HALO_SIZE);
    }
    
    // 计算节点：设置 GPU
    if (pid < np - 1) {
        int deviceCount;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        
        if (err != cudaSuccess) {
            printf("进程 %d: CUDA 不可用: %s\n", pid, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // 轮询分配 GPU
        int device = pid % deviceCount;
        err = cudaSetDevice(device);
        
        if (err != cudaSuccess) {
            printf("进程 %d: 设置 CUDA 设备 %d 失败: %s\n", pid, device, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // 计算节点执行模板计算
        int local_dimz = dimz / (np - 1);
        compute_node_stencil(dimx, dimy, local_dimz, nreps);
        
    } else {
        // 数据服务器
        data_server(dimx, dimy, dimz, nreps);
    }
    
    // 结束 MPI
    MPI_Finalize();
    
    return 0;
}
