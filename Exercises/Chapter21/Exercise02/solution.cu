#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "solution.h"

// ============================================================================
// 设备函数：检查停止条件
// ============================================================================

__device__ bool check_num_points_and_depth(
    Quadtree_node& node, Points* points, int num_points, Parameters params) {
    
    if (params.depth >= params.max_depth || 
        num_points <= params.min_points_per_node) {
        // 停止递归，确保 points[0] 包含所有点
        if (params.point_selector == 1) {
            int it = node.points_begin();
            int end = node.points_end();
            for (it += threadIdx.x; it < end; it += blockDim.x) {
                points[0].set_point(it, points[1].get_point(it));
            }
        }
        return true;
    }
    return false;
}

// ============================================================================
// 设备函数：统计每个象限的点数
// ============================================================================

__device__ void count_points_in_children(
    const Points& in_points, int* smem, 
    int range_begin, int range_end, float2 center) {
    
    // 初始化共享内存
    if (threadIdx.x < 4) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // 统计点数
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);
        
        if (p.x < center.x && p.y >= center.y) {
            atomicAdd(&smem[0], 1);  // 左上
        }
        if (p.x >= center.x && p.y >= center.y) {
            atomicAdd(&smem[1], 1);  // 右上
        }
        if (p.x < center.x && p.y < center.y) {
            atomicAdd(&smem[2], 1);  // 左下
        }
        if (p.x >= center.x && p.y < center.y) {
            atomicAdd(&smem[3], 1);  // 右下
        }
    }
    __syncthreads();
}

// ============================================================================
// 设备函数：计算重排偏移
// ============================================================================

__device__ void scan_for_offsets(int node_points_begin, int* smem) {
    int* smem2 = &smem[4];
    
    if (threadIdx.x == 0) {
        smem2[0] = node_points_begin;
        smem2[1] = smem2[0] + smem[0];
        smem2[2] = smem2[1] + smem[1];
        smem2[3] = smem2[2] + smem[2];
    }
    __syncthreads();
}

// ============================================================================
// 设备函数：重排点
// ============================================================================

__device__ void reorder_points(
    Points* out_points, const Points& in_points, int* smem,
    int range_begin, int range_end, float2 center) {
    
    int* smem2 = &smem[4];
    
    for (int iter = range_begin + threadIdx.x; iter < range_end; iter += blockDim.x) {
        float2 p = in_points.get_point(iter);
        int dest;
        
        if (p.x < center.x && p.y >= center.y) {
            dest = atomicAdd(&smem2[0], 1);  // Top-left
        } else if (p.x >= center.x && p.y >= center.y) {
            dest = atomicAdd(&smem2[1], 1);  // Top-right
        } else if (p.x < center.x && p.y < center.y) {
            dest = atomicAdd(&smem2[2], 1);  // Bottom-left
        } else {
            dest = atomicAdd(&smem2[3], 1);  // Bottom-right
        }
        
        out_points->set_point(dest, p);
    }
    __syncthreads();
}

// ============================================================================
// 设备函数：准备子节点
// ============================================================================

__device__ void prepare_children(
    Quadtree_node* children, Quadtree_node& node, 
    const Bounding_box& bbox, int* smem) {
    
    if (threadIdx.x == 0) {
        const float2& p_min = bbox.get_min();
        const float2& p_max = bbox.get_max();
        
        float2 center;
        bbox.compute_center(&center);
        
        int* smem2 = &smem[4];
        
        for (int i = 0; i < 4; i++) {
            if (smem[i] > 0) {
                children[i].set_id(i);
                children[i].set_range(smem2[i], smem2[i] + smem[i]);
                
                if (i == 0) {  // 左上
                    children[i].set_bounding_box(p_min.x, center.y, center.x, p_max.y);
                } else if (i == 1) {  // 右上
                    children[i].set_bounding_box(center.x, center.y, p_max.x, p_max.y);
                } else if (i == 2) {  // 左下
                    children[i].set_bounding_box(p_min.x, p_min.y, center.x, center.y);
                } else {  // 右下
                    children[i].set_bounding_box(center.x, p_min.y, p_max.x, center.y);
                }
            }
        }
    }
    __syncthreads();
}

// ============================================================================
// 核心 Kernel：递归构建四叉树
// ============================================================================

__global__ void build_quadtree_kernel(
    Quadtree_node* nodes, Points* points, Parameters params) {
    
    __shared__ int smem[8];  // 4个象限点数 + 4个偏移
    
    // The current node in the quadtree
    Quadtree_node& node = nodes[blockIdx.x];
    int num_points = node.num_points();
    
    // 检查停止条件
    if (check_num_points_and_depth(node, points, num_points, params)) {
        return;
    }
    
    // 计算边界框中心
    const Bounding_box& bbox = node.bounding_box();
    float2 center;
    bbox.compute_center(&center);
    
    // 点范围
    int range_begin = node.points_begin();
    int range_end = node.points_end();
    const Points& in_points = points[params.point_selector];
    Points* out_points = &points[(params.point_selector + 1) % 2];
    
    // 统计每个象限点数
    count_points_in_children(in_points, smem, range_begin, range_end, center);
    
    // 计算重排偏移
    scan_for_offsets(node.points_begin(), smem);
    
    // 重排点
    reorder_points(out_points, in_points, smem, range_begin, range_end, center);
    
    // 递归启动子 kernel
    if (threadIdx.x == blockDim.x - 1) {
        // The children
        Quadtree_node* children = 
            &nodes[params.num_nodes_at_this_level + blockIdx.x * 4];
        
        // Prepare children launch
        prepare_children(children, node, bbox, smem);
        
        // Launch 4 children blocks
        build_quadtree_kernel<<<4, blockDim.x, 8 * sizeof(int)>>>(
            children, points, Parameters(params, true));
    }
}

// ============================================================================
// 主机包装函数
// ============================================================================

int build_quadtree(float* h_x, float* h_y, int num_points,
                   int max_depth, int min_points_per_node,
                   float** result_x, float** result_y,
                   float* bounds, int* num_result_points) {
    
    // 分配设备内存
    float *d_x[2], *d_y[2];
    cudaMalloc(&d_x[0], num_points * sizeof(float));
    cudaMalloc(&d_y[0], num_points * sizeof(float));
    cudaMalloc(&d_x[1], num_points * sizeof(float));
    cudaMalloc(&d_y[1], num_points * sizeof(float));
    
    // 复制输入点
    cudaMemcpy(d_x[0], h_x, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y[0], h_y, num_points * sizeof(float), cudaMemcpyHostToDevice);
    
    // 创建 Points 对象
    Points h_points[2];
    h_points[0].set(d_x[0], d_y[0]);
    h_points[1].set(d_x[1], d_y[1]);
    
    Points* d_points;
    cudaMalloc(&d_points, 2 * sizeof(Points));
    cudaMemcpy(d_points, h_points, 2 * sizeof(Points), cudaMemcpyHostToDevice);
    
    // 计算最大节点数
    int max_nodes = 1;
    for (int i = 1; i <= max_depth; i++) {
        max_nodes += (int)pow(4, i);
    }
    max_nodes *= 2;  // 安全余量
    
    // 分配节点内存
    Quadtree_node* d_nodes;
    cudaMalloc(&d_nodes, max_nodes * sizeof(Quadtree_node));
    cudaMemset(d_nodes, 0, max_nodes * sizeof(Quadtree_node));
    
    // 初始化根节点
    Quadtree_node root;
    root.set_id(0);
    root.set_range(0, num_points);
    root.set_bounding_box(bounds[0], bounds[1], bounds[2], bounds[3]);
    cudaMemcpy(d_nodes, &root, sizeof(Quadtree_node), cudaMemcpyHostToDevice);
    
    // 创建参数
    Parameters params(max_depth, min_points_per_node);
    
    // 启动 kernel
    build_quadtree_kernel<<<1, 32>>>(d_nodes, d_points, params);
    
    // 等待完成
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // 复制结果
    *result_x = (float*)malloc(num_points * sizeof(float));
    *result_y = (float*)malloc(num_points * sizeof(float));
    
    cudaMemcpy(*result_x, d_x[0], num_points * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*result_y, d_y[0], num_points * sizeof(float), cudaMemcpyDeviceToHost);
    
    *num_result_points = num_points;
    
    // 清理
    cudaFree(d_x[0]);
    cudaFree(d_y[0]);
    cudaFree(d_x[1]);
    cudaFree(d_y[1]);
    cudaFree(d_points);
    cudaFree(d_nodes);
    
    return 0;
}
