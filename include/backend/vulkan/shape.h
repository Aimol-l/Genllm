#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct ReshapeImpl;
    template <Device D> struct PermuteImpl;
    template <Device D> struct ConcatImpl;
    template <Device D> struct RepeatImpl;
    template <Device D> struct SplitImpl;

    template <>
    struct ReshapeImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct PermuteImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct ConcatImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct RepeatImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SplitImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct ReshapeImpl<Device::VULKAN>;
    extern template struct PermuteImpl<Device::VULKAN>;
    extern template struct ConcatImpl<Device::VULKAN>;
    extern template struct RepeatImpl<Device::VULKAN>;
    extern template struct SplitImpl<Device::VULKAN>;

}
