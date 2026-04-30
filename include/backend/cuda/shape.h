#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ReshapeImpl;
    template <Device D> struct PermuteImpl;
    template <Device D> struct ConcatImpl;
    template <Device D> struct RepeatImpl;

    template <>
    struct ReshapeImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct PermuteImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct ConcatImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct RepeatImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct ReshapeImpl<Device::CUDA>;
    extern template struct PermuteImpl<Device::CUDA>;
    extern template struct ConcatImpl<Device::CUDA>;
    extern template struct RepeatImpl<Device::CUDA>;

}
