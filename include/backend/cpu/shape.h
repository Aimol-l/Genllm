#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ReshapeImpl;
    template <Device D> struct PermuteImpl;
    template <Device D> struct ConcatImpl;
    template <Device D> struct RepeatImpl;
    template <Device D> struct SplitImpl;

    template <>
    struct ReshapeImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct PermuteImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct ConcatImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct RepeatImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SplitImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct ReshapeImpl<Device::CPU>;
    extern template struct PermuteImpl<Device::CPU>;
    extern template struct ConcatImpl<Device::CPU>;
    extern template struct RepeatImpl<Device::CPU>;
    extern template struct SplitImpl<Device::CPU>;

}
