#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct LinearImpl;
    template <Device D> struct MatmulImpl;

    template <>
    struct LinearImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct MatmulImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct LinearImpl<Device::CUDA>;
    extern template struct MatmulImpl<Device::CUDA>;
}