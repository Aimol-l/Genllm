#include <stdexcept>
#include "backend/cuda/normalization.h"

namespace ops {

    void RmsNormImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::rms_norm not implemented"); }
    void LayerNormImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::layer_norm not implemented"); }

template struct RmsNormImpl<Device::CUDA>;
template struct LayerNormImpl<Device::CUDA>;
}
