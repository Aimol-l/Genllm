
#ifdef BACKEND_VULKAN
#include "utils/dtype_traits.hpp"
#include "backend/vulkan/activation.h"

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
    Tensor* in = out->src[0];

    uint64_t total_elems = out->num_elements();
    uint32_t group_x = (total_elems + 255) / 256;
    uint32_t group_y = 1;
    uint32_t group_z = 1;

    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {in, out});

    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);
    ctx.pushConstants(dev_id, &total_elems, sizeof(uint64_t));
    ctx.dispatch(dev_id, group_x, group_y, group_z);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
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

void SigmoidImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    throw std::runtime_error("Vulkan SigmoidImpl: not implemented");
}

template struct SiluImpl<Device::VULKAN>;
template struct GeluImpl<Device::VULKAN>;
template struct ReluImpl<Device::VULKAN>;
template struct SigmoidImpl<Device::VULKAN>;

}

#endif
