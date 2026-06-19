
#ifdef BACKEND_VULKAN

#include "utils/tools.hpp"

#include "utils/dtype_traits.hpp"
#include "backend/vulkan/normalization.h"

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/rms_norm.h"
#include "backend/vulkan/spv/layer_norm.h"
#include "backend/vulkan/push_constants.h"

namespace ops {

static void dispatch_rms_norm(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    Tensor* src = out->src[0];      //bf16
    Tensor* weight = out->src[1];   //fp32

    size_t hidden_size = weight->num_elements();
    int32_t rows = static_cast<int32_t>(src->num_elements() / hidden_size);
    
    NormPushConstants pc{rows, static_cast<int32_t>(hidden_size), out->op_params[0]};

    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {src,weight,out});
    
    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);

    uint32_t group_x = (static_cast<uint32_t>(rows) + 7) / 8;

    ctx.pushConstants(dev_id, &pc, sizeof(NormPushConstants));
    ctx.dispatch(dev_id, group_x, 1, 1);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
}

static void dispatch_layer_norm(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    Tensor* src = out->src[0];
    Tensor* weight = out->src[1];
    Tensor* bias = out->src[2];

    int32_t has_bias = bias ? 1 : 0;
    size_t hidden_size = weight->num_elements();
    int32_t rows = static_cast<int32_t>(src->num_elements() / hidden_size);

    LayerNormPushConstants pc{rows, static_cast<int32_t>(hidden_size), out->op_params[0],has_bias};

    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {src,weight,bias,out});

    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);

    uint32_t group_x = (static_cast<uint32_t>(rows) + 7) / 8;

    ctx.pushConstants(dev_id, &pc, sizeof(LayerNormPushConstants));
    ctx.dispatch(dev_id, group_x, 1, 1);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
}

void RmsNormImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_rms_norm(ctx, dev_id, "rms_norm_f16", vkspv::rms_norm_f16_spv, vkspv::rms_norm_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_rms_norm(ctx, dev_id, "rms_norm_bf16", vkspv::rms_norm_bf16_spv, vkspv::rms_norm_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_rms_norm(ctx, dev_id, "rms_norm_f32", vkspv::rms_norm_f32_spv, vkspv::rms_norm_f32_spv_len, out);
        }
    });
}

void LayerNormImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_layer_norm(ctx, dev_id, "layer_norm_f16", vkspv::layer_norm_f16_spv, vkspv::layer_norm_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_layer_norm(ctx, dev_id, "layer_norm_bf16", vkspv::layer_norm_bf16_spv, vkspv::layer_norm_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_layer_norm(ctx, dev_id, "layer_norm_f32", vkspv::layer_norm_f32_spv, vkspv::layer_norm_f32_spv_len, out);
        }
    });
}

template struct RmsNormImpl<Device::VULKAN>;
template struct LayerNormImpl<Device::VULKAN>;

}

#endif
