#include "core/tensor.hpp"
#include "core/operator.h"

#include "backend/cpu/silu.h"


inline void silu_fp32_kernel(Tensor* output, const Tensor* input){
    // todo..
}



void SiluImpl<Backend::CPU>::execute(Tensor* tensor, const OpContext& ctx){
    if (!tensor->src[0]) {
        throw std::runtime_error("CPU SiLU: missing input tensor");
    }
    switch(tensor->src[0]->dtype){
        case DataType::GGML_TYPE_F32: silu_fp32_kernel(tensor, tensor->src[0]);
        default: throw std::runtime_error("Unsupported data type for cpu silu");
    }
}

template struct SiluImpl<Backend::CPU>;

