#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ReshapeImpl;
    template <Device D> struct PermuteImpl;
    template <Device D> struct ConcatImpl;
    template <Device D> struct RepeatImpl;
    template <Device D> struct SplitImpl;

    template <>
    struct ReshapeImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct PermuteImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct ConcatImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct RepeatImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SplitImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct ReshapeImpl<Device::CUDA>;
    extern template struct PermuteImpl<Device::CUDA>;
    extern template struct ConcatImpl<Device::CUDA>;
    extern template struct RepeatImpl<Device::CUDA>;
    extern template struct SplitImpl<Device::CUDA>;

}
