#include <format>
#include <stdexcept>
#include "backend/cpu/sampling.h"

namespace cpu {

void sampling(Tensor* out) {
    // TODO: sample next token from logits
    // src[0]: logits [vocab_size] (F32)
    // out: scalar int32 token id
    // op_params[0]: temperature
    // op_params[1]: top_k
    // op_params[2]: top_p
    //
    // steps:
    //   1. apply temperature: logits /= temperature
    //   2. optional top-k: keep only top-k logits
    //   3. optional top-p (nucleus): cumulative prob threshold
    //   4. softmax to get probabilities
    //   5. sample from distribution (or argmax if temperature=0)
    throw std::runtime_error(std::format("cpu::sampling not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cpu
