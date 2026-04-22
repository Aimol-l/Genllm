#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct LinearImpl;
    template <Device D> struct MatmulImpl;
    template <Device D> struct TransposeImpl;

    template <>
    struct LinearImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct MatmulImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct TransposeImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct LinearImpl<Device::CUDA>;
    extern template struct MatmulImpl<Device::CUDA>;
    extern template struct TransposeImpl<Device::CUDA>;

}
