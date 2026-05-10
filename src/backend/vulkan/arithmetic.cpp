#include "backend/vulkan/arithmetic.h"
#include "utils/dtype_traits.hpp"

#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/spv/add.h"
#include "backend/vulkan/spv/sub.h"
#include "backend/vulkan/spv/mul.h"
#include "backend/vulkan/spv/div.h"
#include "backend/vulkan/vulkan_context.h"

namespace ops {

static void dispatch_binop(
    VulkanContext& ctx,
    int dev_id,
    const char* opname, const uint32_t* spv, size_t spv_len,
    Tensor* out)
{
    auto& pipe = ctx.getOrCreatePipeline(dev_id, opname, spv, spv_len, 3, sizeof(uint32_t));

    Tensor* src0 = out->src[0];
    Tensor* src1 = out->src[1];

    vk::Buffer buf0 = reinterpret_cast<VkBuffer>(src0->device_handle);
    vk::Buffer buf1 = reinterpret_cast<VkBuffer>(src1->device_handle);
    vk::Buffer buf_dst = reinterpret_cast<VkBuffer>(out->device_handle);

    auto ds = ctx.allocateDescriptorSet(dev_id, pipe.ds_layout);

    vk::DescriptorBufferInfo src0_info(buf0, src0->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo src1_info(buf1, src1->offset, VK_WHOLE_SIZE);
    vk::DescriptorBufferInfo dst_info(buf_dst, out->offset, VK_WHOLE_SIZE);
    ctx.updateDescriptorSets(dev_id, ds, {src0_info, src1_info, dst_info});

    uint64_t total = static_cast<uint64_t>(out->num_elements());

    auto cmd = ctx.beginCommandBuffer(dev_id);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe.pipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipe.layout, 0, ds, {});
    cmd.pushConstants(pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint64_t), &total);
    cmd.dispatch((total + 255) / 256, 1, 1);
    ctx.endSubmitAndWait(dev_id, cmd);

    ctx.freeDescriptorSet(dev_id, ds);
}

void AddImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            dispatch_binop(ctx,dev_id,"add_f16",vkspv::add_f16_spv,vkspv::add_f16_spv_len,out);
        } else if constexpr (std::is_same_v<T,bfloat16_t>) {
            dispatch_binop(ctx,dev_id,"add_bf16",vkspv::add_bf16_spv,vkspv::add_bf16_spv_len,out);
        } else if constexpr (std::is_same_v<T,float>) {
            dispatch_binop(ctx,dev_id,"add_f32",vkspv::add_f32_spv,vkspv::add_f32_spv_len,out);
        }
    });
}

void SubImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            dispatch_binop(ctx,dev_id,"sub_f16",vkspv::sub_f16_spv,vkspv::sub_f16_spv_len,out);
        } else if constexpr (std::is_same_v<T,bfloat16_t>) {
            dispatch_binop(ctx,dev_id,"sub_bf16",vkspv::sub_bf16_spv,vkspv::sub_bf16_spv_len,out);
        } else if constexpr (std::is_same_v<T,float>) {
            dispatch_binop(ctx,dev_id,"sub_f32",vkspv::sub_f32_spv,vkspv::sub_f32_spv_len,out);
        }
    });
}

void MulImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            dispatch_binop(ctx,dev_id,"mul_f16",vkspv::mul_f16_spv,vkspv::mul_f16_spv_len,out);
        } else if constexpr (std::is_same_v<T,bfloat16_t>) {
            dispatch_binop(ctx,dev_id,"mul_bf16",vkspv::mul_bf16_spv,vkspv::mul_bf16_spv_len,out);
        } else if constexpr (std::is_same_v<T,float>) {
            dispatch_binop(ctx,dev_id,"mul_f32",vkspv::mul_f32_spv,vkspv::mul_f32_spv_len,out);
        }
    });
}

void DivImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto& ctx = VulkanContext::get();
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            dispatch_binop(ctx,dev_id,"div_f16",vkspv::div_f16_spv,vkspv::div_f16_spv_len,out);
        } else if constexpr (std::is_same_v<T,bfloat16_t>) {
            dispatch_binop(ctx,dev_id,"div_bf16",vkspv::div_bf16_spv,vkspv::div_bf16_spv_len,out);
        } else if constexpr (std::is_same_v<T,float>) {
            dispatch_binop(ctx,dev_id,"div_f32",vkspv::div_f32_spv,vkspv::div_f32_spv_len,out);
        }
    });
}

template struct AddImpl<Device::VULKAN>;
template struct SubImpl<Device::VULKAN>;
template struct MulImpl<Device::VULKAN>;
template struct DivImpl<Device::VULKAN>;

}

#endif
