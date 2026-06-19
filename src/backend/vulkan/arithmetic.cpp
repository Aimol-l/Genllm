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

static void dispatch_binop(VulkanContext& ctx,int dev_id,const char* name,const uint32_t* spv, size_t spv_len,Tensor* out){
    
    Tensor* src0 = out->src[0];
    Tensor* src1 = out->src[1];

    uint64_t total_elems = out->num_elements();
    uint32_t group_x = (total_elems + 255) / 256;
    uint32_t group_y = 1;
    uint32_t group_z = 1;

    vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {src0, src1,out});

    ctx.bindPipeline(dev_id, name);
    ctx.bindDescriptorSet(dev_id, descSet);
    ctx.pushConstants(dev_id, &total_elems, sizeof(uint64_t));
    ctx.dispatch(dev_id, group_x, group_y, group_z);
    ctx.deferFreeDescriptorSet(dev_id, descSet);
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
