#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct SoftmaxImpl;

    template <>
    struct SoftmaxImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };
    extern template struct SoftmaxImpl<Device::CUDA>;

}
