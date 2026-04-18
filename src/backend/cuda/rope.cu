#include <format>
#include <stdexcept>
#include "backend/cuda/rope.cuh"
#include <cuda_runtime.h>

namespace cuda {

void apply_rope(Tensor* out) {
    // TODO: apply rotary position embedding (CUDA kernel)
    // src[0]: input [batch, n_heads, seq, head_dim] (device ptr)
    // src[1]: cos_cache, src[2]: sin_cache (device ptrs)
    // op_params[0]: n_past
    // tip: 1 thread per element, load cos/sin from cache, apply rotation
    //       for BF16: use __bfloat162 for vectorized load
    throw std::runtime_error(std::format("cuda::apply_rope not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void rope_cache(Tensor* out) {
    // TODO: precompute cos/sin tables (CUDA kernel or host + cudaMemcpy)
    // op_params[0]: theta, op_params[1]: head_dim, op_params[2]: max_seq_len
    // tip: can compute on host then cudaMemcpy, or launch kernel on device
    throw std::runtime_error(std::format("cuda::rope_cache not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
