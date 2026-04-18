#pragma once
#include "core/tensor.hpp"

namespace kernel {

    void add(Tensor* out);
    void sub(Tensor* out);
    void mul(Tensor* out);
    void div(Tensor* out);
    void scale(Tensor* out);

    void rms_norm(Tensor* out);
    void layer_norm(Tensor* out);

    void matmul(Tensor* out);
    void linear(Tensor* out);
    void transpose(Tensor* out);

    void reshape(Tensor* out);
    void permute(Tensor* out);

    void silu(Tensor* out);
    void gelu(Tensor* out);
    void relu(Tensor* out);

    void softmax(Tensor* out);
    void diag_mask_inf(Tensor* out);

    void embedding(Tensor* out);
    void apply_rope(Tensor* out);
    
    void sdpa(Tensor* out);
    void attention(Tensor* out);
    void flash_attention(Tensor* out);

    void concat(Tensor* out);
    void repeat(Tensor* out);

    void rope_cache(Tensor* out);
    void sampling(Tensor* out);
    
}
