#include "backend/vulkan/linear.h"
#include "utils/dtype_traits.hpp"


#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/spv/linear.h"
#include "backend/vulkan/push_constants.h"

namespace ops {

static void dispatch_gemv(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    Tensor* x = out->src[0];
    Tensor* w = out->src[1];
    int32_t M = static_cast<int32_t>(x->dims[1]);
    int32_t K = static_cast<int32_t>(x->dims[2]);
    int32_t N = static_cast<int32_t>(w->dims[0]);

    LinearPushConstants pc{M, K, N};
    // warp-level: local_size=256, subgroup=32 → 8 outputs per workgroup
    constexpr uint32_t WG_OUT = 256 / 32;
    uint32_t group_x = (static_cast<uint32_t>(N) + WG_OUT - 1) / WG_OUT;

    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {x, w, out});
    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);
    ctx.pushConstants(dev_id, &pc, sizeof(LinearPushConstants));
    ctx.dispatch(dev_id, group_x, 1, 1);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
}

static void dispatch_linear(
    VulkanContext& ctx, int dev_id,
    const char* name, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    Tensor* x = out->src[0]; // [1,M,K]
    Tensor* w = out->src[1]; //   [N,K]
    int32_t M = static_cast<int32_t>(x->dims[1]);
    int32_t K = static_cast<int32_t>(x->dims[2]);
    int32_t N = static_cast<int32_t>(w->dims[0]);

    LinearPushConstants pc{M,K,N};
    static constexpr uint32_t TILE = 16;
    
    uint32_t group_x = (static_cast<uint32_t>(N) + TILE - 1) / TILE;
    uint32_t group_y = (static_cast<uint32_t>(M) + TILE - 1) / TILE;
    uint32_t group_z = 1;


    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {x, w,out});
    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);

    ctx.pushConstants(dev_id, &pc, sizeof(LinearPushConstants));

    ctx.dispatch(dev_id, group_x, group_y, group_z);

    ctx.deferFreeDescriptorSet(dev_id, descSet);

}

void LinearImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    Tensor* x = out->src[0];
    int32_t M = static_cast<int32_t>(x->dims[1]);
    // 小 M（主要是 decode 的 M=1）走 GEMV 快速路径，避免 cooperative matrix 的浪费
    if (M <= 4) {
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            if constexpr (std::is_same_v<T, float16_t>) {
                dispatch_gemv(ctx, dev_id, "linear_gemv_f16", vkspv::linear_gemv_f16_spv, vkspv::linear_gemv_f16_spv_len, out);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                dispatch_gemv(ctx, dev_id, "linear_gemv_bf16", vkspv::linear_gemv_bf16_spv, vkspv::linear_gemv_bf16_spv_len, out);
            } else if constexpr (std::is_same_v<T, float>) {
                dispatch_gemv(ctx, dev_id, "linear_gemv_f32", vkspv::linear_gemv_f32_spv, vkspv::linear_gemv_f32_spv_len, out);
            }
        });
    } else {
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            if constexpr (std::is_same_v<T, float16_t>) {
                dispatch_linear(ctx, dev_id, "linear_f16", vkspv::linear_f16_spv, vkspv::linear_f16_spv_len, out);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                dispatch_linear(ctx, dev_id, "linear_bf16", vkspv::linear_bf16_spv, vkspv::linear_bf16_spv_len, out);
            } else if constexpr (std::is_same_v<T, float>) {
                dispatch_linear(ctx, dev_id, "linear_f32", vkspv::linear_f32_spv, vkspv::linear_f32_spv_len, out);
            }
        });
    }
}

void MatmulImpl<Device::VULKAN>::execute(Tensor*, int32_t) {
    throw std::runtime_error("Vulkan MatmulImpl: not implemented");
}

template struct MatmulImpl<Device::VULKAN>;
template struct LinearImpl<Device::VULKAN>;

}

#endif
