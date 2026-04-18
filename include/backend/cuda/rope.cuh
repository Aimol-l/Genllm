#pragma once
#include "core/tensor.hpp"

namespace cuda {

// apply rotary position embedding to q or k
// src[0]: input [batch, n_heads, seq, head_dim]
// src[1]: cos_cache, src[2]: sin_cache
// op_params[0]: n_past
void apply_rope(Tensor* out);

// precompute cos/sin tables for RoPE
// op_params[0]: theta, op_params[1]: head_dim, op_params[2]: max_seq_len
void rope_cache(Tensor* out);

} // namespace cuda
