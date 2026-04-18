#include <format>
#include <stdexcept>
#include "backend/cuda/embedding.cuh"
#include <cuda_runtime.h>

namespace cuda {

void embedding(Tensor* out) {
    // TODO: embedding lookup (CUDA kernel)
    // src[0]: input_ids I32 [batch, seq_len] (device ptr)
    // src[1]: weight [vocab_size, hidden_size] (device ptr)
    // out = weight[input_ids, :]
    // tip: 1 thread per (batch, seq, hidden), gather from weight using input_ids
    //       or use cuSPARSELt if weight is quantized
    throw std::runtime_error(std::format("cuda::embedding not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void get_rows(Tensor* out) {
    // TODO: alias for embedding
    embedding(out);
}

} // namespace cuda
