#pragma once
#include "core/tensor.hpp"


namespace ops {

    template <Device D> struct RmsNormImpl;
    template <Device D> struct LayerNormImpl;

    template <>
    struct RmsNormImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct LayerNormImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct RmsNormImpl<Device::CUDA>;
    extern template struct LayerNormImpl<Device::CUDA>;

}
