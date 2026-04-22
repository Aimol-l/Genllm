#include <stdexcept>
#include "backend/cuda/sampling.h"

namespace ops {

    void SamplingImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::sampling not implemented"); }

template struct SamplingImpl<Device::CUDA>;
}
