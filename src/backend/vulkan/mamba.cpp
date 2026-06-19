
#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "utils/dtype_traits.hpp"
#include "backend/vulkan/mamba.h"
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/push_constants.h"

namespace ops {

void CausalConv1dImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    // TODO: implement causal conv1d
    throw std::runtime_error("CausalConv1dImpl<VULKAN>: not implemented");
}

void SsmScanImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    // TODO: implement ssm scan
    throw std::runtime_error("SsmScanImpl<VULKAN>: not implemented");
}

template struct CausalConv1dImpl<Device::VULKAN>;
template struct SsmScanImpl<Device::VULKAN>;

}

#endif
