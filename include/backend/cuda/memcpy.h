#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct MemcpyImpl;

    template <>
    struct MemcpyImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct MemcpyImpl<Device::CUDA>;
}

