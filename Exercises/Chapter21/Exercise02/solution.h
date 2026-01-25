#ifndef CHAPTER_21_EXERCISE02_SOLUTION_H
#define CHAPTER_21_EXERCISE02_SOLUTION_H

#include <cuda_runtime.h>

// ============================================================================
// 2D 点集合类
// ============================================================================

class Points {
    float* m_x;
    float* m_y;

public:
    __host__ __device__ Points() : m_x(NULL), m_y(NULL) {}
    __host__ __device__ Points(float* x, float* y) : m_x(x), m_y(y) {}
    
    __host__ __device__ __forceinline__ float2 get_point(int idx) const {
        return make_float2(m_x[idx], m_y[idx]);
    }
    
    __host__ __device__ __forceinline__ void set_point(int idx, const float2& p) {
        m_x[idx] = p.x;
        m_y[idx] = p.y;
    }
    
    __host__ __device__ __forceinline__ void set(float* x, float* y) {
        m_x = x;
        m_y = y;
    }
};

// ============================================================================
// 2D 边界框类
// ============================================================================

class Bounding_box {
    float2 m_p_min;
    float2 m_p_max;

public:
    __host__ __device__ Bounding_box() {
        m_p_min = make_float2(0.0f, 0.0f);
        m_p_max = make_float2(1.0f, 1.0f);
    }
    
    __host__ __device__ void compute_center(float2* center) const {
        center->x = 0.5f * (m_p_min.x + m_p_max.x);
        center->y = 0.5f * (m_p_min.y + m_p_max.y);
    }
    
    __host__ __device__ __forceinline__ const float2& get_max() const { return m_p_max; }
    __host__ __device__ __forceinline__ const float2& get_min() const { return m_p_min; }
    
    __host__ __device__ bool contains(const float2& p) const {
        return p.x >= m_p_min.x && p.x < m_p_max.x && 
               p.y >= m_p_min.y && p.y < m_p_max.y;
    }
    
    __host__ __device__ void set(float min_x, float min_y, float max_x, float max_y) {
        m_p_min.x = min_x;
        m_p_min.y = min_y;
        m_p_max.x = max_x;
        m_p_max.y = max_y;
    }
};

// ============================================================================
// 四叉树节点类
// ============================================================================

class Quadtree_node {
    int m_id;
    // The bounding box of the tree
    Bounding_box m_bounding_box;
    // The range of points
    int m_begin, m_end;

public:
    __host__ __device__ Quadtree_node() : m_id(0), m_begin(0), m_end(0) {}
    
    __host__ __device__ int id() const { return m_id; }
    __host__ __device__ void set_id(int new_id) { m_id = new_id; }
    
    __host__ __device__ __forceinline__ const Bounding_box& bounding_box() const {
        return m_bounding_box;
    }
    
    __host__ __device__ __forceinline__ void set_bounding_box(
        float min_x, float min_y, float max_x, float max_y) {
        m_bounding_box.set(min_x, min_y, max_x, max_y);
    }
    
    __host__ __device__ __forceinline__ int num_points() const { return m_end - m_begin; }
    __host__ __device__ __forceinline__ int points_begin() const { return m_begin; }
    __host__ __device__ __forceinline__ int points_end() const { return m_end; }
    
    __host__ __device__ __forceinline__ void set_range(int begin, int end) {
        m_begin = begin;
        m_end = end;
    }
};

// ============================================================================
// 算法参数结构
// ============================================================================

struct Parameters {
    int point_selector;           // 双缓冲选择器
    int num_nodes_at_this_level;  // 当前层节点数
    int depth;                    // 当前深度
    const int max_depth;          // 最大深度
    const int min_points_per_node;// 停止阈值
    
    __host__ __device__ Parameters(int max_d, int min_p)
        : point_selector(0), num_nodes_at_this_level(1), depth(0),
          max_depth(max_d), min_points_per_node(min_p) {}
    
    __host__ __device__ Parameters(const Parameters& params, bool)
        : point_selector((params.point_selector + 1) % 2),
          num_nodes_at_this_level(4 * params.num_nodes_at_this_level),
          depth(params.depth + 1),
          max_depth(params.max_depth),
          min_points_per_node(params.min_points_per_node) {}
};

// ============================================================================
// 函数声明
// ============================================================================

/**
 * 构建四叉树
 * @param h_x, h_y     输入点坐标
 * @param num_points   点数量
 * @param max_depth    最大深度
 * @param min_points   节点最小点数
 * @param result_x/y   输出重排后的点
 * @param bounds       边界框 [min_x, min_y, max_x, max_y]
 * @param num_result   结果点数
 * @return 0=成功
 */
int build_quadtree(float* h_x, float* h_y, int num_points,
                   int max_depth, int min_points_per_node,
                   float** result_x, float** result_y,
                   float* bounds, int* num_result_points);

#endif // CHAPTER_21_EXERCISE02_SOLUTION_H
