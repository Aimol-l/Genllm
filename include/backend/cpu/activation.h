#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct SiluImpl;
    template <Device D> struct GeluImpl;
    template <Device D> struct ReluImpl;
    template <Device D> struct SigmoidImpl;

    template <>
    struct SiluImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct GeluImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct ReluImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SigmoidImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct SiluImpl<Device::CPU>;
    extern template struct GeluImpl<Device::CPU>;
    extern template struct ReluImpl<Device::CPU>;
    extern template struct SigmoidImpl<Device::CPU>;

}
