#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <cuda_runtime.h>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }
    
    // 返回经过的时间（毫秒）
    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return duration.count() / 1000.0;
    }
    
    // 返回经过的时间（微秒）
    double elapsed_us() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return static_cast<double>(duration.count());
    }
    
    // 打印经过的时间
    void print(const char* label = "Elapsed time") const {
        printf("%s: %.3f ms\n", label, elapsed_ms());
    }
};

// CUDA 事件计时器（更精确）
class CudaTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    
public:
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event, 0);
    }
    
    void stop() {
        cudaEventRecord(stop_event, 0);
        cudaEventSynchronize(stop_event);
    }
    
    // 返回经过的时间（毫秒）
    float elapsed_ms() const {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
    
    // 打印经过的时间
    void print(const char* label = "CUDA Elapsed time") const {
        printf("%s: %.3f ms\n", label, elapsed_ms());
    }
};

#endif // TIMER_H
