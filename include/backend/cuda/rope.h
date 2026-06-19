#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ApplyRopeImpl;
    template <Device D> struct RopeCacheImpl;

    template <>
    struct ApplyRopeImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct RopeCacheImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct ApplyRopeImpl<Device::CUDA>;
    extern template struct RopeCacheImpl<Device::CUDA>;

}
