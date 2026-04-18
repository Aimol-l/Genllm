#pragma once
#include "core/tensor.hpp"

namespace cuda {

// out = softmax(src[0], dim=op_params[0])
void softmax(Tensor* out);

// fill upper-triangular with -inf for causal masking
// src[0]: scores [seq, seq]
void diag_mask_inf(Tensor* out);

// flash attention (fused, memory-efficient)
// src[0]: Q, src[1]: K, src[2]: V
// op_params[0]: scale, op_params[1]: causal flag
void sdpa(Tensor* out);
void attention(Tensor* out);
void flash_attention(Tensor* out);

} // namespace cuda
