#include <format>
#include <stdexcept>
#include "backend/cuda/normalization.cuh"
#include <cuda_runtime.h>

namespace cuda {

void rms_norm(Tensor* out) {
    // TODO: RMS normalization (CUDA kernel)
    // src[0]: input (device ptr), src[1]: weight (device ptr)
    // out = x / sqrt(mean(x^2) + eps) * weight
    // tip: use warp-level reduction for variance, then grid-level
    //       or call cuBLAS for partial reduction
    throw std::runtime_error(std::format("cuda::rms_norm not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void layer_norm(Tensor* out) {
    // TODO: layer normalization (CUDA kernel)
    // src[0]: input, src[1]: weight, src[2]: bias (all device ptrs)
    // out = (x - mean) / sqrt(var + eps) * weight + bias
    // tip: two-pass reduction (mean then var), or use cuBLAS rms_norm + manual mean
    throw std::runtime_error(std::format("cuda::layer_norm not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
