#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <cublas_v2.h>
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "backend/cuda/transpose.h"
#include "utils/dtype_traits.hpp"


namespace ops {

void TransposeImpl<Device::CUDA>::execute(Tensor*, int32_t) { 
    throw std::runtime_error("cuda::transpose not implemented"); 
}

template struct TransposeImpl<Device::CUDA>;

}
