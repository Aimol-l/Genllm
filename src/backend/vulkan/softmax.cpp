
#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>

#include "core/manager.h"
#include "utils/dtype_traits.hpp"
#include "core/page_attention.h"
#include "backend/vulkan/softmax.h"
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/push_constants.h"


namespace ops {


void SoftmaxImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan SoftmaxImpl: not implemented");
}


template struct SoftmaxImpl<Device::VULKAN>;

}

#endif
