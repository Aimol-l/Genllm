#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct LinearImpl;
    template <Device D> struct MatmulImpl;
    template <Device D> struct TransposeImpl;


    template <>
    struct LinearImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct MatmulImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    template <>
    struct TransposeImpl<Device::CPU> {
        static void execute(Tensor* out);
    };

    extern template struct LinearImpl<Device::CPU>;
    extern template struct MatmulImpl<Device::CPU>;
    extern template struct TransposeImpl<Device::CPU>;
}