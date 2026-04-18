#pragma once
#include "core/tensor.hpp"

namespace cuda {

// out = view of src[0] with new shape (no data copy)
void reshape(Tensor* out);

// out = view of src[0] with axes permuted (no data copy)
// op_params[0..3]: permutation order
void permute(Tensor* out);

// out = concatenation of src[0], src[1] along axis op_params[0]
void concat(Tensor* out);

// out = src[0] repeated along dimension
// op_params[0]: repeats, op_params[1]: axis
void repeat(Tensor* out);

} // namespace cuda
