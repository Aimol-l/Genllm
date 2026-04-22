#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ApplyRopeImpl;
    template <Device D> struct RopeCacheImpl;

    template <>
    struct ApplyRopeImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct RopeCacheImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct ApplyRopeImpl<Device::CUDA>;
    extern template struct RopeCacheImpl<Device::CUDA>;

}
