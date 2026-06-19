
#ifdef BACKEND_VULKAN

#include "backend/vulkan/rope.h"
#include "utils/dtype_traits.hpp"

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/rope.h"
#include "backend/vulkan/push_constants.h"

namespace ops {

static void dispatch_rope(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    Tensor* src = out->src[0];
    Tensor* cos = out->src[1];
    Tensor* sin = out->src[2];

    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {src, cos,sin,out});

    int32_t head_dim  = static_cast<int32_t>(out->op_params[0]);
    int32_t half_dim  = head_dim / 2;
    int32_t start_pos = static_cast<int32_t>(out->op_params[2]);
    int32_t B       = static_cast<int32_t>(src->dims[0]);
    int32_t n_heads = static_cast<int32_t>(src->dims[1]);
    int32_t seq_len = static_cast<int32_t>(src->dims[2]);

    RopePushConstants pc{head_dim, half_dim, seq_len, start_pos, n_heads, B};

    int64_t n_pairs = static_cast<int64_t>(B) * n_heads * seq_len * half_dim;
    uint32_t group_x = (static_cast<uint32_t>(n_pairs) + 255) / 256;

    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);
    ctx.pushConstants(dev_id, &pc, sizeof(RopePushConstants));
    ctx.dispatch(dev_id, group_x, 1, 1);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
}

void ApplyRopeImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_rope(ctx, dev_id, "rope_f16", vkspv::rope_f16_spv, vkspv::rope_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_rope(ctx, dev_id, "rope_bf16", vkspv::rope_bf16_spv, vkspv::rope_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_rope(ctx, dev_id, "rope_f32", vkspv::rope_f32_spv, vkspv::rope_f32_spv_len, out);
        }
    });
}

template struct ApplyRopeImpl<Device::VULKAN>;

}

#endif
