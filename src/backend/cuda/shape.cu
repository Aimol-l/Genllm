#include <format>
#include <stdexcept>
#include "backend/cuda/shape.cuh"
#include <cuda_runtime.h>

namespace cuda {

void reshape(Tensor* out) {
    // TODO: reshape (view only, no data copy)
    // same as CPU version: just update dims/strides, share device data pointer
    throw std::runtime_error(std::format("cuda::reshape not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void permute(Tensor* out) {
    // TODO: permute axes (view only, no data copy)
    // op_params[0..n-1]: permutation order
    throw std::runtime_error(std::format("cuda::permute not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void concat(Tensor* out) {
    // TODO: concatenate along axis (CUDA kernel)
    // src[0], src[1]: device ptrs, op_params[0]: axis
    // tip: simple memcpy-based kernel, one block per output row
    throw std::runtime_error(std::format("cuda::concat not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void repeat(Tensor* out) {
    // TODO: repeat along dimension (CUDA kernel)
    // op_params[0]: repeats, op_params[1]: axis
    throw std::runtime_error(std::format("cuda::repeat not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
