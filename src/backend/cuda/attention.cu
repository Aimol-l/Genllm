#include <format>
#include <stdexcept>
#include "backend/cuda/attention.cuh"
#include <cuda_runtime.h>

namespace cuda {

void softmax(Tensor* out) {
    // TODO: softmax (CUDA kernel)
    // op_params[0]: axis (default: last dim)
    // tip: warp-level reduce for max & sum, use __shfl_down_sync
    //       or call cuDNN softmax
    throw std::runtime_error(std::format("cuda::softmax not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void diag_mask_inf(Tensor* out) {
    // TODO: causal diagonal mask (CUDA kernel)
    // set positions where j > i to -inf
    // tip: 1 thread per element, simple condition check
    throw std::runtime_error(std::format("cuda::diag_mask_inf not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void sdpa(Tensor* out) {
    // TODO: scaled dot-product attention (CUDA kernel)
    // src[0]: Q, src[1]: K, src[2]: V
    // op_params[0]: scale, op_params[1]: causal flag
    // out = softmax(Q @ K^T * scale + mask) @ V
    // tip: use Flash Attention or xFormers for production
    //       naive: cublasGemmBatched for Q@K^T, softmax kernel, cublasGemmBatched for @V
    throw std::runtime_error(std::format("cuda::sdpa not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void flash_attn(Tensor* out) {
    // TODO: flash attention (CUDA kernel, memory-efficient)
    // tip: use FlashAttention-2 / FlashInfer library
    //       tiling: load K,V blocks from HBM, online softmax, write O in blocks
    //       support F16/BF16 natively, Q4 dequantize on-the-fly
    throw std::runtime_error(std::format("cuda::flash_attn not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
