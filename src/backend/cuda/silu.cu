#include "core/tensor.hpp"
#include "core/operator.h"

#include "backend/cuda/silu.h"


__global__ void silu_fp32_kernel(Tensor* output, const Tensor* input){
    // todo..
}



void SiluImpl<Backend::CUDA>::execute(Tensor* tensor, const OpContext& ctx){
    if (!tensor->src[0]) {
        throw std::runtime_error("CUDA SiLU: missing input tensor");
    }
    switch(tensor->src[0]->dtype){
        case DataType::GGML_TYPE_F32: silu_fp32_kernel(tensor, tensor->src[0]);
        default: throw std::runtime_error("Unsupported data type for CUDA silu");
    }
}

template struct SiluImpl<Backend::CUDA>;

