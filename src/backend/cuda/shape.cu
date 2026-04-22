#include <stdexcept>
#include "backend/cuda/shape.h"

namespace ops {

    void ReshapeImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::reshape not implemented"); }
    void PermuteImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::permute not implemented"); }
    void ConcatImpl<Device::CUDA>::execute(Tensor*)  { throw std::runtime_error("cuda::concat not implemented"); }
    void RepeatImpl<Device::CUDA>::execute(Tensor*)  { throw std::runtime_error("cuda::repeat not implemented"); }

template struct ReshapeImpl<Device::CUDA>;
template struct PermuteImpl<Device::CUDA>;
template struct ConcatImpl<Device::CUDA>;
template struct RepeatImpl<Device::CUDA>;
}
