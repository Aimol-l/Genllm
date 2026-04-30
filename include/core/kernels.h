#pragma once
#include "core/tensor.hpp"

#ifdef _WIN32
    #define OPS_API __declspec(dllexport)
    #define NOMINMAX 1 // prevent windows redefining min/max
#else
    #define OPS_API // Linux or macOS
#endif

namespace kernel {

    OPS_API void add(Tensor* out, int32_t dev_id = 0);
    OPS_API void sub(Tensor* out, int32_t dev_id = 0);
    OPS_API void mul(Tensor* out, int32_t dev_id = 0);
    OPS_API void div(Tensor* out, int32_t dev_id = 0);
    OPS_API void rms_norm(Tensor* out, int32_t dev_id = 0);
    OPS_API void layer_norm(Tensor* out, int32_t dev_id = 0);
    OPS_API void matmul(Tensor* out, int32_t dev_id = 0);
    OPS_API void linear(Tensor* out, int32_t dev_id = 0);
    OPS_API void transpose(Tensor* out, int32_t dev_id = 0);
    OPS_API void reshape(Tensor* out, int32_t dev_id = 0);
    OPS_API void permute(Tensor* out, int32_t dev_id = 0);
    OPS_API void silu(Tensor* out, int32_t dev_id = 0);
    OPS_API void gelu(Tensor* out, int32_t dev_id = 0);
    OPS_API void relu(Tensor* out, int32_t dev_id = 0);
    OPS_API void softmax(Tensor* out, int32_t dev_id = 0);
    OPS_API void diag_mask_inf(Tensor* out, int32_t dev_id = 0);
    OPS_API void embedding(Tensor* out, int32_t dev_id = 0);
    OPS_API void apply_rope(Tensor* out, int32_t dev_id = 0);
    OPS_API void sdpa(Tensor* out, int32_t dev_id = 0);
    OPS_API void attention(Tensor* out, int32_t dev_id = 0);
    OPS_API void flash_attention(Tensor* out, int32_t dev_id = 0);
    OPS_API void concat(Tensor* out, int32_t dev_id = 0);
    OPS_API void repeat(Tensor* out, int32_t dev_id = 0);
    OPS_API void memcpy(Tensor* out, int32_t dev_id = 0);
    OPS_API void rope_cache(Tensor* out, int32_t dev_id = 0);
    
} // namespace kernel