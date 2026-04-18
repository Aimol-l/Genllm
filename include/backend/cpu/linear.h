#pragma once
#include "core/tensor.hpp"

namespace cpu {

// out = src[0] @ src[1]  (matrix multiplication, C = A @ B^T or A @ B)
// src[0]: [M, K], src[1]: [K, N] or [N, K], out: [M, N]
void matmul(Tensor* out);

// out = src[0] @ weight^T + bias  (fully-connected layer)
// src[0]: input [batch, in_features], src[1]: weight [out_features, in_features], src[2]: bias [out_features]
void linear(Tensor* out);

// out[i][j] = src[i][j]  with dims transposed (e.g. [M, K] -> [K, M])
// op_params[0], op_params[1]: the two axes to swap
void transpose(Tensor* out);

} // namespace cpu
