#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct AddImpl;
    template <Device D> struct SubImpl;
    template <Device D> struct MulImpl;
    template <Device D> struct DivImpl;
    template <Device D> struct ScaleImpl;

    template <>
    struct AddImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct SubImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct MulImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct DivImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    template <>
    struct ScaleImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct AddImpl<Device::CUDA>;
    extern template struct SubImpl<Device::CUDA>;
    extern template struct MulImpl<Device::CUDA>;
    extern template struct DivImpl<Device::CUDA>;
    extern template struct ScaleImpl<Device::CUDA>;

}
