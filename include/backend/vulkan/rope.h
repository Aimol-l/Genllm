#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ApplyRopeImpl;
    template <Device D> struct RopeCacheImpl;

    template <>
    struct ApplyRopeImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct RopeCacheImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct ApplyRopeImpl<Device::VULKAN>;
    extern template struct RopeCacheImpl<Device::VULKAN>;

}
