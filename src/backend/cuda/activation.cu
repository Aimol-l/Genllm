#include <format>
#include <stdexcept>
#include "backend/cuda/activation.cuh"
#include <cuda_runtime.h>

namespace cuda {

void silu(Tensor* out) {
    // TODO: SiLU activation (CUDA kernel)
    // out = x * sigmoid(x)
    // tip: use __expf for fast sigmoid, or cuda::silu from cuDNN
    throw std::runtime_error(std::format("cuda::silu not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void gelu(Tensor* out) {
    // TODO: GELU activation (CUDA kernel)
    // out = x * 0.5 * (1 + erf(x / sqrt(2)))
    // tip: erff() is available in CUDA math, or use tanh approximation
    throw std::runtime_error(std::format("cuda::gelu not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void relu(Tensor* out) {
    // TODO: ReLU activation (CUDA kernel)
    // out = max(0, x)
    // tip: trivial kernel, or use cuDNN activation
    throw std::runtime_error(std::format("cuda::relu not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
