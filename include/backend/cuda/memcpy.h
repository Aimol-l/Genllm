#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct MemcpyImpl;

    template <>
    struct MemcpyImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct MemcpyImpl<Device::CUDA>;
}

