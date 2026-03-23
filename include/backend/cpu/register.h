// cpu_ops.h - CPU 算子实现
#pragma once
#include "cpu/silu.h"
#include "core/operator.h"

namespace {

    struct CpuKernelInitializer {
        CpuKernelInitializer() {
            // ==================== 显式注册函数 ====================
            register_op(SiluImpl<Backend::CPU>::execute,OperationType::OP_TYPE_SILU,"silu",Backend::CPU);
        }
    };
    static CpuKernelInitializer g_cpu_kernel_init;
}
