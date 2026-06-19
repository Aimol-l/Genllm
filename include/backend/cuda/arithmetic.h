#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct AddImpl;
    template <Device D> struct SubImpl;
    template <Device D> struct MulImpl;
    template <Device D> struct DivImpl;

    template <>
    struct AddImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SubImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct MulImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct DivImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct AddImpl<Device::CUDA>;
    extern template struct SubImpl<Device::CUDA>;
    extern template struct MulImpl<Device::CUDA>;
    extern template struct DivImpl<Device::CUDA>;

}
