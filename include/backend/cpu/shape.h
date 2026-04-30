#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ReshapeImpl;
    template <Device D> struct PermuteImpl;
    template <Device D> struct ConcatImpl;
    template <Device D> struct RepeatImpl;

    template <>
    struct ReshapeImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct PermuteImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct ConcatImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct RepeatImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    extern template struct ReshapeImpl<Device::CPU>;
    extern template struct PermuteImpl<Device::CPU>;
    extern template struct ConcatImpl<Device::CPU>;
    extern template struct RepeatImpl<Device::CPU>;

}
