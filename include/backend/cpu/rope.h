#pragma once
#include "core/tensor.hpp"

namespace cpu {

// apply rotary position embedding to q or k
// src[0]: input [batch, n_heads, seq, head_dim]
// src[1]: cos_cache [max_seq, half_dim]
// src[2]: sin_cache [max_seq, half_dim]
// op_params[0]: number of past positions (for KV cache offset)
void apply_rope(Tensor* out);

// precompute cos/sin tables for RoPE
// out->data: float [max_seq, half_dim]
// op_params[0]: theta (base freq, e.g. 10000.0)
// op_params[1]: head_dim
// op_params[2]: max_seq_len
void rope_cache(Tensor* out);

} // namespace cpu
