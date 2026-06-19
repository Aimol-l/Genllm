#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct AddImpl;
    template <Device D> struct SubImpl;
    template <Device D> struct MulImpl;
    template <Device D> struct DivImpl;

    template <>
    struct AddImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SubImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct MulImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct DivImpl<Device::CPU> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct AddImpl<Device::CPU>;
    extern template struct SubImpl<Device::CPU>;
    extern template struct MulImpl<Device::CPU>;
    extern template struct DivImpl<Device::CPU>;

}
