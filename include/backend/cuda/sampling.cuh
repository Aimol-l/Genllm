#pragma once
#include "core/tensor.hpp"

namespace cuda {

// sample next token from logits
// src[0]: logits [vocab_size], out: scalar int32
// op_params[0]: temperature, op_params[1]: top_k, op_params[2]: top_p
void sampling(Tensor* out);

} // namespace cuda
