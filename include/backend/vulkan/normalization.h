#pragma once
#include "core/tensor.hpp"


namespace ops {

    template <Device D> struct RmsNormImpl;
    template <Device D> struct LayerNormImpl;

    template <>
    struct RmsNormImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct LayerNormImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct RmsNormImpl<Device::VULKAN>;
    extern template struct LayerNormImpl<Device::VULKAN>;

}
