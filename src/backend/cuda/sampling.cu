#include <format>
#include <stdexcept>
#include "backend/cuda/sampling.cuh"
#include <cuda_runtime.h>

namespace cuda {

void sampling(Tensor* out) {
    // TODO: sample next token (CUDA kernel)
    // src[0]: logits [vocab_size] (device ptr, F32)
    // out: scalar int32 token id (can be host or device)
    // op_params[0]: temperature, op_params[1]: top_k, op_params[2]: top_p
    // tip: softmax + top-k/top-p filtering on GPU, then
    //       cudaMemcpy the filtered probs to host for random sampling,
    //       or use cuRAND for GPU-side sampling
    throw std::runtime_error(std::format("cuda::sampling not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
