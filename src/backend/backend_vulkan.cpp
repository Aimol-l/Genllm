#include <cstdint>
#include "backend/backend.h"


#ifdef BACKEND_VULKAN
#include <print>
#include <vulkan/vulkan.hpp>
#include "backend/vulkan/spv/add.h"
#include "backend/vulkan/spv/sub.h"
#include "backend/vulkan/spv/mul.h"
#include "backend/vulkan/spv/div.h"
#include "backend/vulkan/spv/silu.h"
#include "backend/vulkan/spv/gelu.h"
#include "backend/vulkan/spv/relu.h"
#include "backend/vulkan/spv/linear.h"
#include "backend/vulkan/spv/rms_norm.h"
#include "backend/vulkan/spv/layer_norm.h"
#include "backend/vulkan/spv/rope.h"
#include "backend/vulkan/spv/permute.h"
#include "backend/vulkan/spv/page_attention.h"
#include "backend/vulkan/spv/flash_attention.h"
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/push_constants.h"

VulkanBackendProvider::VulkanBackendProvider() {
    try {
        ctx_ = &VulkanContext::get();
        available_ = ctx_->device_count() > 0;
    } catch (const std::exception& e) {
        std::println("VulkanBackendProvider: initialization failed: {}", e.what());
        available_ = false;
    }
}

bool VulkanBackendProvider::is_available() const { 
    return available_; 
}
int VulkanBackendProvider::get_device_count() const { 
    return available_ ? ctx_->device_count() : 0;
 }

BackendInfo VulkanBackendProvider::get_backend_info(int device_id) const {
    BackendInfo info;
    info.id = static_cast<size_t>(device_id);
    info.device = Device::VULKAN;

    if (!available_) return info;

    auto phy = ctx_->physical_device(device_id);
    auto mem_props = phy.getMemoryProperties();

    size_t total = 0;
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        if (mem_props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            total += mem_props.memoryHeaps[i].size;
        }
    }

    info.total_memory = total;
    info.used_memory = 0;
    info.compute_power = 1.0;
    info.bandwidth = 32.0;
    return info;
}

void VulkanBackendProvider::print_device_info(int device_id) const {
    if (!available_) return;

    auto phy = ctx_->physical_device(device_id);
    auto props = phy.getProperties();
    auto mem_props = phy.getMemoryProperties();

    std::println("  Device Name:       {}", std::string_view(props.deviceName));
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; ++i) {
        auto heap = mem_props.memoryHeaps[i];
        bool is_device_local = (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{};
        std::println("  Memory Heap[{}]:    {} MB ({})",
            i, heap.size / (1024 * 1024),
            is_device_local ? "DeviceLocal (VRAM)" : "HostVisible (System RAM)");
    }
    std::println("  Max Work Group Size:    [{}, {}, {}]",
        props.limits.maxComputeWorkGroupSize[0],
        props.limits.maxComputeWorkGroupSize[1],
        props.limits.maxComputeWorkGroupSize[2]);
    std::println("  Max Work Group Count:    [{}, {}, {}]",
        props.limits.maxComputeWorkGroupCount[0],
        props.limits.maxComputeWorkGroupCount[1],
        props.limits.maxComputeWorkGroupCount[2]);
    std::println("  Max Work Group Invocations:    {}", props.limits.maxComputeWorkGroupInvocations);

    auto print_cooperative_matrix_info = [&]() {
        auto inst = ctx_->instance();
        auto fn = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            inst.getProcAddr("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (!fn) {
            std::println("  Cooperative Matrix: not supported");
            return;
        }

        uint32_t count = 0;
        VkResult result = fn(static_cast<VkPhysicalDevice>(phy), &count, nullptr);
        if (result != VK_SUCCESS || count == 0) {
            std::println("  Cooperative Matrix: no configurations");
            return;
        }

        std::vector<VkCooperativeMatrixPropertiesKHR> cm_props(count);
        for (auto& p : cm_props) {
            p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            p.pNext = nullptr;
        }
        fn(static_cast<VkPhysicalDevice>(phy), &count, cm_props.data());

        std::println("Cooperative Matrix (KHR)");
        for (size_t i = 0; i < cm_props.size(); ++i) {
            auto& p = cm_props[i];
            std::println("    [{:0>2}]: {:>13}[{:>2},{:>2}] @ {:>13}[{:>2},{:>2}] + {:>7}[{:>2},{:>2}] = {:>7}[{:>2},{:>2}],scope={},satAcc={}",
                i,
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.AType)),
                p.MSize, p.KSize,
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.BType)),
                p.KSize, p.NSize,
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.CType)),
                p.MSize, p.NSize,
                vk::to_string(static_cast<vk::ComponentTypeKHR>(p.ResultType)),
                p.MSize, p.NSize,
                vk::to_string(static_cast<vk::ScopeKHR>(p.scope)),
                static_cast<bool>(p.saturatingAccumulation)
            );
        }
    };
    print_cooperative_matrix_info(); // A[m,k] @ B[k,n] + C[m,n] = D[m,n]
}

static struct VulkanBackendProviderRegistrar {
    VulkanBackendProviderRegistrar() {
        auto provider = std::make_unique<VulkanBackendProvider>();
        if (provider->is_available()) {
            BackendRegistry::instance().register_provider(std::move(provider));
        }
        // 预先注册所有算子
        auto& instance = VulkanContext::get();

        // --- 算术 ---
        instance.registerOp("add_f16",  vkspv::add_f16_spv,  vkspv::add_f16_spv_len,  3, sizeof(uint64_t));
        instance.registerOp("add_bf16", vkspv::add_bf16_spv, vkspv::add_bf16_spv_len, 3, sizeof(uint64_t));
        instance.registerOp("add_f32",  vkspv::add_f32_spv,  vkspv::add_f32_spv_len,  3, sizeof(uint64_t));
        instance.registerOp("sub_f16",  vkspv::sub_f16_spv,  vkspv::sub_f16_spv_len,  3, sizeof(uint64_t));
        instance.registerOp("sub_bf16", vkspv::sub_bf16_spv, vkspv::sub_bf16_spv_len, 3, sizeof(uint64_t));
        instance.registerOp("sub_f32",  vkspv::sub_f32_spv,  vkspv::sub_f32_spv_len,  3, sizeof(uint64_t));
        instance.registerOp("mul_f16",  vkspv::mul_f16_spv,  vkspv::mul_f16_spv_len,  3, sizeof(uint64_t));
        instance.registerOp("mul_bf16", vkspv::mul_bf16_spv, vkspv::mul_bf16_spv_len, 3, sizeof(uint64_t));
        instance.registerOp("mul_f32",  vkspv::mul_f32_spv,  vkspv::mul_f32_spv_len,  3, sizeof(uint64_t));
        instance.registerOp("div_f16",  vkspv::div_f16_spv,  vkspv::div_f16_spv_len,  3, sizeof(uint64_t));
        instance.registerOp("div_bf16", vkspv::div_bf16_spv, vkspv::div_bf16_spv_len, 3, sizeof(uint64_t));
        instance.registerOp("div_f32",  vkspv::div_f32_spv,  vkspv::div_f32_spv_len,  3, sizeof(uint64_t));

        // --- 激活 ---
        instance.registerOp("silu_f16",  vkspv::silu_f16_spv,  vkspv::silu_f16_spv_len,  2, sizeof(uint64_t));
        instance.registerOp("silu_bf16", vkspv::silu_bf16_spv, vkspv::silu_bf16_spv_len, 2, sizeof(uint64_t));
        instance.registerOp("silu_f32",  vkspv::silu_f32_spv,  vkspv::silu_f32_spv_len,  2, sizeof(uint64_t));
        instance.registerOp("gelu_f16",  vkspv::gelu_f16_spv,  vkspv::gelu_f16_spv_len,  2, sizeof(uint64_t));
        instance.registerOp("gelu_bf16", vkspv::gelu_bf16_spv, vkspv::gelu_bf16_spv_len, 2, sizeof(uint64_t));
        instance.registerOp("gelu_f32",  vkspv::gelu_f32_spv,  vkspv::gelu_f32_spv_len,  2, sizeof(uint64_t));
        instance.registerOp("relu_f16",  vkspv::relu_f16_spv,  vkspv::relu_f16_spv_len,  2, sizeof(uint64_t));
        instance.registerOp("relu_bf16", vkspv::relu_bf16_spv, vkspv::relu_bf16_spv_len, 2, sizeof(uint64_t));
        instance.registerOp("relu_f32",  vkspv::relu_f32_spv,  vkspv::relu_f32_spv_len,  2, sizeof(uint64_t));

        // --- 线性 ---
        instance.registerOp("linear_f16",  vkspv::linear_f16_spv,  vkspv::linear_f16_spv_len,  3, sizeof(ops::LinearPushConstants));
        instance.registerOp("linear_bf16", vkspv::linear_bf16_spv, vkspv::linear_bf16_spv_len, 3, sizeof(ops::LinearPushConstants));
        instance.registerOp("linear_f32",  vkspv::linear_f32_spv,  vkspv::linear_f32_spv_len,  3, sizeof(ops::LinearPushConstants));
        instance.registerOp("linear_gemv_f16",  vkspv::linear_gemv_f16_spv,  vkspv::linear_gemv_f16_spv_len,  3, sizeof(ops::LinearPushConstants));
        instance.registerOp("linear_gemv_bf16", vkspv::linear_gemv_bf16_spv, vkspv::linear_gemv_bf16_spv_len, 3, sizeof(ops::LinearPushConstants));
        instance.registerOp("linear_gemv_f32",  vkspv::linear_gemv_f32_spv,  vkspv::linear_gemv_f32_spv_len,  3, sizeof(ops::LinearPushConstants));

        // --- RMS Norm ---
        instance.registerOp("rms_norm_f16",  vkspv::rms_norm_f16_spv,  vkspv::rms_norm_f16_spv_len,  3, sizeof(ops::NormPushConstants));
        instance.registerOp("rms_norm_bf16", vkspv::rms_norm_bf16_spv, vkspv::rms_norm_bf16_spv_len, 3, sizeof(ops::NormPushConstants));
        instance.registerOp("rms_norm_f32",  vkspv::rms_norm_f32_spv,  vkspv::rms_norm_f32_spv_len,  3, sizeof(ops::NormPushConstants));

        // --- Layer Norm ---
        instance.registerOp("layer_norm_f16",  vkspv::layer_norm_f16_spv,  vkspv::layer_norm_f16_spv_len,  4, sizeof(ops::LayerNormPushConstants));
        instance.registerOp("layer_norm_bf16", vkspv::layer_norm_bf16_spv, vkspv::layer_norm_bf16_spv_len, 4, sizeof(ops::LayerNormPushConstants));
        instance.registerOp("layer_norm_f32",  vkspv::layer_norm_f32_spv,  vkspv::layer_norm_f32_spv_len,  4, sizeof(ops::LayerNormPushConstants));

        // --- RoPE ---
        instance.registerOp("rope_f16",  vkspv::rope_f16_spv,  vkspv::rope_f16_spv_len,  4, sizeof(ops::RopePushConstants));
        instance.registerOp("rope_bf16", vkspv::rope_bf16_spv, vkspv::rope_bf16_spv_len, 4, sizeof(ops::RopePushConstants));
        instance.registerOp("rope_f32",  vkspv::rope_f32_spv,  vkspv::rope_f32_spv_len,  4, sizeof(ops::RopePushConstants));

        // --- Permute ---
        instance.registerOp("permute_f16",  vkspv::permute_f16_spv,  vkspv::permute_f16_spv_len,  2, sizeof(ops::PermutePushConstants));
        instance.registerOp("permute_bf16", vkspv::permute_bf16_spv, vkspv::permute_bf16_spv_len, 2, sizeof(ops::PermutePushConstants));
        instance.registerOp("permute_f32",  vkspv::permute_f32_spv,  vkspv::permute_f32_spv_len,  2, sizeof(ops::PermutePushConstants));

        // --- Flash Attention ---
        instance.registerOp("flash_attn_f16",  vkspv::flash_attn_f16_spv,  vkspv::flash_attn_f16_spv_len,  4, sizeof(ops::FlashPushConstants));
        instance.registerOp("flash_attn_bf16", vkspv::flash_attn_bf16_spv, vkspv::flash_attn_bf16_spv_len, 4, sizeof(ops::FlashPushConstants));

        // --- Paged Attention ---
        instance.registerOp("page_attn_f16",  vkspv::page_attn_f16_spv,  vkspv::page_attn_f16_spv_len,  6, sizeof(ops::PagedAttnPushConstants));
        instance.registerOp("page_attn_bf16", vkspv::page_attn_bf16_spv, vkspv::page_attn_bf16_spv_len, 6, sizeof(ops::PagedAttnPushConstants));
    }
} g_vulkan_backend_registrar;

#endif // BACKEND_VULKAN
