#pragma once
#include "utils/utils.hpp"

struct Tensor;
class OpContext;


template <Backend D>
struct SiluImpl;

template <>
struct SiluImpl<Backend::CUDA> {
    static void execute(Tensor* tensor, const OpContext& ctx);
};


extern template struct SiluImpl<Backend::CUDA>;
