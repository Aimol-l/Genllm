#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct PagedAttentionImpl;
    template <Device D> struct FlashAttentionImpl;

    template <>
    struct PagedAttentionImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct FlashAttentionImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct PagedAttentionImpl<Device::CPU>;
    extern template struct FlashAttentionImpl<Device::CPU>;

}
