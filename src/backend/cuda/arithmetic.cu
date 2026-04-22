#include <stdexcept>
#include "backend/cuda/arithmetic.h"

namespace ops {

    void AddImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::add not implemented"); }
    void SubImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::sub not implemented"); }
    void MulImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::mul not implemented"); }
    void DivImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::div not implemented"); }
    void ScaleImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::scale not implemented"); }

template struct AddImpl<Device::CUDA>;
template struct SubImpl<Device::CUDA>;
template struct MulImpl<Device::CUDA>;
template struct DivImpl<Device::CUDA>;
template struct ScaleImpl<Device::CUDA>;
}
