

#ifdef BACKEND_VULKAN

#include "backend/vulkan/shape.h"
#include "utils/dtype_traits.hpp"

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/permute.h"
#include "backend/vulkan/push_constants.h"

namespace ops {

static void dispatch_permute(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out, const PermutePushConstants& pc)
{
    Tensor* x = out->src[0];
    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {x, out});

    uint32_t group_x = (static_cast<uint32_t>(pc.total) + 255) / 256;

    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);
    ctx.pushConstants(dev_id, &pc, sizeof(PermutePushConstants));
    ctx.dispatch(dev_id, group_x, 1, 1);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
}

void ReshapeImpl<Device::VULKAN>::execute(Tensor* out, int32_t) {
    const Tensor* x = out->src[0];
    out->device_handle = x->device_handle;
    out->offset        = x->offset;

    size_t stride  = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (out->dims[i] == 0) {
            out->strides[i] = 0;
        } else {
            out->strides[i] = stride;
            stride *= static_cast<size_t>(out->dims[i]);
        }
    }
}

void PermuteImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    const Tensor* x = out->src[0];

    int ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) ndim = i + 1;
    if (ndim <= 1) return;

    size_t elem_sz = data_type_size(out->dtype);

    PermutePushConstants pc{};
    pc.ndim  = ndim;
    pc.total = static_cast<int32_t>(out->num_elements());
    for (int i = 0; i < ndim; i++) {
        pc.out_strides[i] = static_cast<int32_t>(out->strides[i]);
        pc.src_strides[i] = static_cast<int32_t>(x->strides[i]);
        pc.perm[i]        = static_cast<int32_t>(out->op_params[i]);
    }

    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_permute(ctx, dev_id, "permute_f16", vkspv::permute_f16_spv, vkspv::permute_f16_spv_len, out, pc);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_permute(ctx, dev_id, "permute_bf16", vkspv::permute_bf16_spv, vkspv::permute_bf16_spv_len, out, pc);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_permute(ctx, dev_id, "permute_f32", vkspv::permute_f32_spv, vkspv::permute_f32_spv_len, out, pc);
        }
    });
}

void ConcatImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan ConcatImpl: not implemented");
}
void RepeatImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan RepeatImpl: not implemented");
}

void SplitImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    throw std::runtime_error("Vulkan SplitImpl: not implemented");
}

template struct ReshapeImpl<Device::VULKAN>;
template struct PermuteImpl<Device::VULKAN>;
template struct ConcatImpl<Device::VULKAN>;
template struct RepeatImpl<Device::VULKAN>;
template struct SplitImpl<Device::VULKAN>;

}

#endif
