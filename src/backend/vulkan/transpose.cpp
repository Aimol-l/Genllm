#include "backend/vulkan/transpose.h"
#include "utils/dtype_traits.hpp"


#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/linear.h"
#include "backend/vulkan/push_constants.h"

namespace ops {

void TransposeImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan TransposeImpl: not implemented");
}

template struct TransposeImpl<Device::VULKAN>;

}

#endif
