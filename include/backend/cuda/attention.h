#pragma once
#include "core/tensor.hpp"
#include <cstdint>

namespace ops {

    template <Device D> struct SoftmaxImpl;
    template <Device D> struct DiagMaskInfImpl;
    template <Device D> struct SdpaImpl;
    template <Device D> struct AttentionImpl;
    template <Device D> struct FlashAttentionImpl;

    template <>
    struct SoftmaxImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct DiagMaskInfImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct SdpaImpl<Device::CUDA> {
        static void execute(Tensor* out,int32_t device_id);
    };

    template <>
    struct AttentionImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct FlashAttentionImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct SoftmaxImpl<Device::CUDA>;
    extern template struct DiagMaskInfImpl<Device::CUDA>;
    extern template struct SdpaImpl<Device::CUDA>;
    extern template struct AttentionImpl<Device::CUDA>;
    extern template struct FlashAttentionImpl<Device::CUDA>;

}
