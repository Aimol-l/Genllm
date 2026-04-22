#include <format>
#include <stdexcept>
#include "backend/cpu/sampling.h"
#include "utils/dtype_traits.hpp"

namespace ops {

    void SamplingImpl<Device::CPU>::execute(Tensor* out) {
        throw std::runtime_error("cpu::sampling not implemented");
    }

template struct SamplingImpl<Device::CPU>;
}
