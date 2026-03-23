// cpu_ops.h - CPU 算子实现
#pragma once
#include "cuda/silu.h"
#include "core/operator.h"

namespace {

    struct CUDAKernelInitializer {
        CUDAKernelInitializer() {
            // ==================== 显式注册函数 ====================
            register_op(SiluImpl<Backend::CUDA>::execute,OperationType::OP_TYPE_SILU,"silu",Backend::CUDA);
        }
    };
    static CUDAKernelInitializer g_cuda_kernel_init;
}
