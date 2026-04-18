#include <format>
#include <stdexcept>
#include "backend/cuda/arithmetic.cuh"
#include <cuda_runtime.h>

namespace cuda {

void add(Tensor* out) {
    // TODO: element-wise add (CUDA kernel)
    // src[0]: input A (device ptr), src[1]: input B (device ptr)
    // out[i] = A[i] + B[i]
    // dtype dispatch: F32/F16/BF16 __half/__nv_bfloat16, quantized need dequant kernel
    throw std::runtime_error(std::format("cuda::add not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void sub(Tensor* out) {
    // TODO: element-wise sub (CUDA kernel)
    throw std::runtime_error(std::format("cuda::sub not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void mul(Tensor* out) {
    // TODO: element-wise mul / broadcast (CUDA kernel)
    throw std::runtime_error(std::format("cuda::mul not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void div(Tensor* out) {
    // TODO: element-wise div / broadcast (CUDA kernel)
    throw std::runtime_error(std::format("cuda::div not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void scale(Tensor* out) {
    // TODO: scalar multiply (CUDA kernel)
    // out[i] = src[0][i] * op_params[0]
    throw std::runtime_error(std::format("cuda::scale not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
