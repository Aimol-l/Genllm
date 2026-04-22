#include <stdexcept>
#include "backend/cuda/activation.h"

namespace ops {

    void SiluImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::silu not implemented"); }
    void GeluImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::gelu not implemented"); }
    void ReluImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::relu not implemented"); }

template struct SiluImpl<Device::CUDA>;
template struct GeluImpl<Device::CUDA>;
template struct ReluImpl<Device::CUDA>;
}
