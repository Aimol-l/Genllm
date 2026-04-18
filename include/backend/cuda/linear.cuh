#pragma once
#include "core/tensor.hpp"

namespace cuda {

// out = src[0] @ src[1]  (matrix multiplication)
// src[0]: [M, K], src[1]: [K, N] or [N, K], out: [M, N]
void matmul(Tensor* out);

// out = src[0] @ weight^T + bias  (fully-connected layer)
// src[0]: input [batch, in_features], src[1]: weight, src[2]: bias
void linear(Tensor* out);

// out[i][j] = src[i][j]  with dims transposed
// op_params[0], op_params[1]: axes to swap
void transpose(Tensor* out);

} // namespace cuda
