#include <stdexcept>
#include "backend/cuda/embedding.h"

namespace ops {

    void EmbeddingImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::embedding not implemented"); }

template struct EmbeddingImpl<Device::CUDA>;
}
