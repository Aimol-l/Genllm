#include <stdexcept>
#include "backend/cuda/rope.h"

namespace ops {

    void ApplyRopeImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::apply_rope not implemented"); }
    void RopeCacheImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::rope_cache not implemented"); }

template struct ApplyRopeImpl<Device::CUDA>;
template struct RopeCacheImpl<Device::CUDA>;
}
