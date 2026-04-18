#pragma once
#include "core/tensor.hpp"

namespace cpu {

// out[batch, seq, hidden] = weight[input_ids[batch, seq], :]
// src[0]: input_ids I32 [batch, seq_len], src[1]: weight [vocab_size, hidden_size]
void embedding(Tensor* out);

} // namespace cpu
