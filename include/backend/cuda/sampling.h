#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct SamplingImpl;

    template <>
    struct SamplingImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct SamplingImpl<Device::CUDA>;

}
