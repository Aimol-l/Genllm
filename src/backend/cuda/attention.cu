#include <stdexcept>
#include "backend/cuda/attention.h"

namespace ops {

    void SoftmaxImpl<Device::CUDA>::execute(Tensor*)         { throw std::runtime_error("cuda::softmax not implemented"); }
    void DiagMaskInfImpl<Device::CUDA>::execute(Tensor*)     { throw std::runtime_error("cuda::diag_mask_inf not implemented"); }
    void SdpaImpl<Device::CUDA>::execute(Tensor*)            { throw std::runtime_error("cuda::sdpa not implemented"); }
    void AttentionImpl<Device::CUDA>::execute(Tensor*)       { throw std::runtime_error("cuda::attention not implemented"); }
    void FlashAttentionImpl<Device::CUDA>::execute(Tensor*)  { throw std::runtime_error("cuda::flash_attention not implemented"); }

template struct SoftmaxImpl<Device::CUDA>;
template struct DiagMaskInfImpl<Device::CUDA>;
template struct SdpaImpl<Device::CUDA>;
template struct AttentionImpl<Device::CUDA>;
template struct FlashAttentionImpl<Device::CUDA>;
}
