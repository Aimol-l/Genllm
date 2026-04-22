#include <stdexcept>
#include "backend/cuda/linear.h"

namespace ops {

    void LinearImpl<Device::CUDA>::execute(Tensor*)    { throw std::runtime_error("cuda::linear not implemented"); }
    void MatmulImpl<Device::CUDA>::execute(Tensor*)    { throw std::runtime_error("cuda::matmul not implemented"); }
    void TransposeImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::transpose not implemented"); }

template struct LinearImpl<Device::CUDA>;
template struct MatmulImpl<Device::CUDA>;
template struct TransposeImpl<Device::CUDA>;
}
