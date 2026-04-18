#pragma once
#include "core/tensor.hpp"

namespace cpu {

// out = softmax(src[0], dim=op_params[0])
// typical: last dim softmax for attention scores
void softmax(Tensor* out);

// fill upper-triangular (above diagonal) with -inf for causal masking
// src[0]: scores [seq, seq], out: masked scores
void diag_mask_inf(Tensor* out);

// out = softmax(Q @ K^T / sqrt(d)) @ V  (scaled dot-product attention)
// src[0]: Q, src[1]: K, src[2]: V
// op_params[0]: scale (1/sqrt(d_k)), op_params[1]: causal mask flag
void sdpa(Tensor* out);
void attention(Tensor* out);
void flash_attention(Tensor* out);

} // namespace cpu
