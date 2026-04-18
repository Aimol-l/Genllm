#pragma once
#include "core/tensor.hpp"

namespace cuda {

// out = x / sqrt(mean(x^2) + eps) * weight
// src[0]: input, src[1]: weight, op_params[0]: eps
void rms_norm(Tensor* out);

// out = (x - mean) / sqrt(var + eps) * weight + bias
// src[0]: input, src[1]: weight, src[2]: bias, op_params[0]: eps
void layer_norm(Tensor* out);

} // namespace cuda
