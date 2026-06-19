#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct CausalConv1dImpl;
    template <Device D> struct SsmScanImpl;

    template <>
    struct CausalConv1dImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SsmScanImpl<Device::VULKAN> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct CausalConv1dImpl<Device::VULKAN>;
    extern template struct SsmScanImpl<Device::VULKAN>;

}
