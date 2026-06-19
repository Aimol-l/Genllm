#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct SiluImpl;
    template <Device D> struct GeluImpl;
    template <Device D> struct ReluImpl;
    template <Device D> struct SigmoidImpl;

    template <>
    struct SiluImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct GeluImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct ReluImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SigmoidImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct SiluImpl<Device::VULKAN>;
    extern template struct GeluImpl<Device::VULKAN>;
    extern template struct ReluImpl<Device::VULKAN>;
    extern template struct SigmoidImpl<Device::VULKAN>;

}
