#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct SiluImpl;
    template <Device D> struct GeluImpl;
    template <Device D> struct ReluImpl;
    template <Device D> struct SigmoidImpl;

    template <>
    struct SiluImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct GeluImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct ReluImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    template <>
    struct SigmoidImpl<Device::CUDA> {
        static void execute(Tensor* out, int32_t dev_id);
    };

    extern template struct SiluImpl<Device::CUDA>;
    extern template struct GeluImpl<Device::CUDA>;
    extern template struct ReluImpl<Device::CUDA>;
    extern template struct SigmoidImpl<Device::CUDA>;

}
