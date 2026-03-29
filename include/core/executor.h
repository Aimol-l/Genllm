#pragma once
#include "graph.hpp"
#include "memory.hpp"
#include <print>

struct ComputeContext {
    void* cuda_stream = nullptr;    // cudaStream_t
    void* sycl_queue = nullptr;     // sycl::queue*
    void* vk_cmd_buffer = nullptr;  // VkCommandBuffer
    int cpu_threads = 4;
};

class Executor {
public:
    void execute(ComputeGraph& graph, ComputeContext& ctx) {
        std::println("=== Execution Started ===");
        
        for (Tensor* t : graph.get_execution_order()) {
            if (t->op_type == OperationType::OP_MEMCPY) {
                execute_memcpy(t, ctx);
            } else {
                dispatch_kernel(t, ctx);
            }
        }
        
        std::println("=== Execution Complete ===");
    }

private:
    void execute_memcpy(Tensor* t, ComputeContext& ctx) {
        Tensor* src = t->src[0];
        void* dst = t->data;
        void* src_ptr = src->data;
        size_t bytes = t->bytes();
        
        // 实际实现：cudaMemcpy / clEnqueueCopy 等
        #ifdef __CUDACC__
        if (src->device != t->device) {
            cudaMemcpyAsync(dst, src_ptr, bytes, 
                           get_cuda_memcpy_kind(src->device, t->device),
                           static_cast<cudaStream_t>(ctx.cuda_stream));
        }
        #else
        memcpy(dst, src_ptr, bytes);  // 简化
        #endif
    }
    
    void dispatch_kernel(Tensor* t, ComputeContext& ctx) {
        switch (t->device) {
            case Device::CPU:
                launch_cpu_kernel(t, ctx);
                break;
            case Device::CUDA_0:
            case Device::CUDA_1:
                launch_cuda_kernel(t, ctx);
                break;
            default:
                std::println("Warning: Unknown device {}", to_string(t->device));
                launch_cpu_kernel(t, ctx);
        }
    }
    
    void launch_cpu_kernel(Tensor* t, ComputeContext& /*ctx*/) {
        // 实际实现：调用 CPU 后端算子
        std::println("  [CPU] {}", t->name);
    }
    
    void launch_cuda_kernel(Tensor* t, ComputeContext& ctx) {
        // 实际实现：调用 CUDA kernel
        std::println("  [CUDA] {}", t->name);
    }
    
    #ifdef __CUDACC__
    cudaMemcpyKind get_cuda_memcpy_kind(Device src, Device dst) {
        if (src == Device::CPU && (dst == Device::CUDA_0 || dst == Device::CUDA_1)) {
            return cudaMemcpyHostToDevice;
        } else if ((src == Device::CUDA_0 || src == Device::CUDA_1) && dst == Device::CPU) {
            return cudaMemcpyDeviceToHost;
        } else {
            return cudaMemcpyDeviceToDevice;
        }
    }
    #endif
};