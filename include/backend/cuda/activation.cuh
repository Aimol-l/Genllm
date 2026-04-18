#pragma once
#include "core/tensor.hpp"

namespace cuda {

// out = x * sigmoid(x)
void silu(Tensor* out);

// out = x * 0.5 * (1 + erf(x / sqrt(2)))
void gelu(Tensor* out);

// out = max(0, x)
void relu(Tensor* out);

} // namespace cuda
