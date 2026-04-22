#pragma once
#include "core/tensor.hpp"

#ifdef _WIN32
    #define OPS_API __declspec(dllexport)
    #define NOMINMAX 1 // prevent windows redefining min/max
#else
    #define OPS_API // Linux or macOS
#endif


namespace kernel {

    OPS_API void add(Tensor* out);
    OPS_API void sub(Tensor* out);
    OPS_API void mul(Tensor* out);
    OPS_API void div(Tensor* out);
    OPS_API void scale(Tensor* out);
    OPS_API void rms_norm(Tensor* out);
    OPS_API void layer_norm(Tensor* out);
    OPS_API void matmul(Tensor* out);
    OPS_API void linear(Tensor* out);
    OPS_API void transpose(Tensor* out);
    OPS_API void reshape(Tensor* out);
    OPS_API void permute(Tensor* out);
    OPS_API void silu(Tensor* out);
    OPS_API void gelu(Tensor* out);
    OPS_API void relu(Tensor* out);
    OPS_API void softmax(Tensor* out);
    OPS_API void diag_mask_inf(Tensor* out);
    OPS_API void embedding(Tensor* out);
    OPS_API void apply_rope(Tensor* out);
    OPS_API void sdpa(Tensor* out);
    OPS_API void attention(Tensor* out);
    OPS_API void flash_attention(Tensor* out);
    OPS_API void concat(Tensor* out);
    OPS_API void repeat(Tensor* out);
    OPS_API void rope_cache(Tensor* out);
    OPS_API void sampling(Tensor* out);

} // namespace kernel