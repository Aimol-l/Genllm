#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct SiluImpl;
    template <Device D> struct GeluImpl;
    template <Device D> struct ReluImpl;

    template <>
    struct SiluImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct GeluImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct ReluImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    extern template struct SiluImpl<Device::CPU>;
    extern template struct GeluImpl<Device::CPU>;
    extern template struct ReluImpl<Device::CPU>;

}
