#pragma once
#include "core/tensor.hpp"

namespace cpu {

// out = view of src[0] with new shape  (no data copy, only metadata change)
// out->dims: target shape, src[0]: input
void reshape(Tensor* out);

// out = view of src[0] with axes permuted  (no data copy, only metadata change)
// op_params[0..3]: permutation order of axes
void permute(Tensor* out);

// out = concatenation of src[0], src[1] along axis op_params[0]
void concat(Tensor* out);

// out = src[0] repeated op_params[0] times along dimension op_params[1]
void repeat(Tensor* out);

} // namespace cpu
