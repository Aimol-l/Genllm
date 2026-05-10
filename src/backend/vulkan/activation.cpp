#include "backend/vulkan/activation.h"
#include "utils/dtype_traits.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/silu.h"
#include "backend/vulkan/spv/gelu.h"
#include "backend/vulkan/spv/relu.h"

namespace ops {

static void dispatch_unary(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, name, spv, spv_len, 2, sizeof(uint64_t));

    Tensor* src = out->src[0];
    vk::Buffer buf_src = reinterpret_cast<VkBuffer>(src->device_handle);
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);

    vk::DescriptorBufferInfo src_info(buf_src, src->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {src_info, dst_info});

    uint64_t total = static_cast<uint64_t>(out->num_elements());

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint64_t), &total);
    cmd.dispatch((total + 255) / 256, 1, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

void SiluImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_unary(ctx, dev_id, "silu_f16", vkspv::silu_f16_spv, vkspv::silu_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_unary(ctx, dev_id, "silu_bf16", vkspv::silu_bf16_spv, vkspv::silu_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_unary(ctx, dev_id, "silu_f32", vkspv::silu_f32_spv, vkspv::silu_f32_spv_len, out);
        }
    });
}

void GeluImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_unary(ctx, dev_id, "gelu_f16", vkspv::gelu_f16_spv, vkspv::gelu_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_unary(ctx, dev_id, "gelu_bf16", vkspv::gelu_bf16_spv, vkspv::gelu_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_unary(ctx, dev_id, "gelu_f32", vkspv::gelu_f32_spv, vkspv::gelu_f32_spv_len, out);
        }
    });
}

void ReluImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            dispatch_unary(ctx, dev_id, "relu_f16", vkspv::relu_f16_spv, vkspv::relu_f16_spv_len, out);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            dispatch_unary(ctx, dev_id, "relu_bf16", vkspv::relu_bf16_spv, vkspv::relu_bf16_spv_len, out);
        } else if constexpr (std::is_same_v<T, float>) {
            dispatch_unary(ctx, dev_id, "relu_f32", vkspv::relu_f32_spv, vkspv::relu_f32_spv_len, out);
        }
    });
}

template struct SiluImpl<Device::VULKAN>;
template struct GeluImpl<Device::VULKAN>;
template struct ReluImpl<Device::VULKAN>;

}

#endif
