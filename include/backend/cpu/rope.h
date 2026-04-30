#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ApplyRopeImpl;
    template <Device D> struct RopeCacheImpl;

    template <>
    struct ApplyRopeImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct RopeCacheImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    extern template struct ApplyRopeImpl<Device::CPU>;
    extern template struct RopeCacheImpl<Device::CPU>;

}
