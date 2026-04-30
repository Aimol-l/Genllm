#pragma once
#include "core/tensor.hpp"


namespace ops {

    template <Device D> struct RmsNormImpl;
    template <Device D> struct LayerNormImpl;

    template <>
    struct RmsNormImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct LayerNormImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    extern template struct RmsNormImpl<Device::CPU>;
    extern template struct LayerNormImpl<Device::CPU>;

}
