#pragma once
#include "core/tensor.hpp"

namespace ops {

    template <Device D> struct EmbeddingImpl;

    template <>
    struct EmbeddingImpl<Device::CUDA> {
        static void execute(Tensor* out);
    };

    extern template struct EmbeddingImpl<Device::CUDA>;

}
