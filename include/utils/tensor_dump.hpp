#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "core/tensor.hpp"

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif
#ifdef BACKEND_VULKAN
#include "backend/vulkan/vulkan_context.h"
#endif

namespace dump {

// 将任何类型的 tensor（BF16/F32/Q4_K/Q6_K...）转换为 float32 数组
// out 由内部 new[] 分配，调用者需 delete[]
// host_data: 可选，提供 host 端数据指针（用于 device tensor）
void dequantize_to_f32(const Tensor* t, float*& out, size_t& num_elems,
                       const void* host_data = nullptr);

// 将 float32 数据保存为 .npy 文件
void save_npy_f32(const std::string& path, const float* data,
                  const int64_t* shape, int ndim);

// 便捷函数：dump tensor → .npy
inline void dump_tensor_npy(const Tensor* t, const std::string& path) {
    const void* host_data = t->data;
    std::unique_ptr<char[]> device_copy;
    if (t->device != Device::CPU) {
        size_t nbytes = t->bytes();
        device_copy = std::make_unique<char[]>(nbytes);
        switch (t->device) {
        case Device::CUDA:
#ifdef BACKEND_CUDA
            cudaMemcpy(device_copy.get(), t->data, nbytes, cudaMemcpyDeviceToHost);
#endif
            break;
        case Device::VULKAN:
#ifdef BACKEND_VULKAN
        {
            auto& vk_ctx = VulkanContext::get();
            int vk_dev = 0;
            vk::DeviceMemory staging_mem;
            void* staging_mapped = nullptr;
            vk::Buffer staging_buf = vk_ctx.createStagingBuffer(vk_dev, nbytes, &staging_mem, &staging_mapped);
            if (staging_buf) {
                auto& slot = vk_ctx.slot(vk_dev);
                vk::CommandBufferAllocateInfo allocInfo(slot.command_pool, vk::CommandBufferLevel::ePrimary, 1);
                auto cmds = slot.device.allocateCommandBuffers(allocInfo);
                vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
                cmds[0].begin(beginInfo);
                cmds[0].copyBuffer(
                    static_cast<VkBuffer>(reinterpret_cast<void*>(t->device_handle)),
                    staging_buf,
                    vk::BufferCopy(t->offset, 0, nbytes));
                cmds[0].end();
                vk::SubmitInfo submitInfo({}, {}, cmds[0]);
                vk::Fence fence = slot.device.createFence({});
                slot.compute_queue.submit(submitInfo, fence);
                VkFence vk_fence = static_cast<VkFence>(fence);
                vkWaitForFences(static_cast<VkDevice>(slot.device), 1, &vk_fence, VK_TRUE, UINT64_MAX);
                slot.device.destroyFence(fence);
                slot.device.freeCommandBuffers(slot.command_pool, cmds[0]);
                std::memcpy(device_copy.get(), staging_mapped, nbytes);
                vk_ctx.destroyStagingBuffer(vk_dev, staging_buf, staging_mem, staging_mapped);
            }
        }
#endif
            break;
        default:
            break;
        }
        host_data = device_copy.get();
    }

    float* data = nullptr;
    size_t n = 0;
    dequantize_to_f32(t, data, n, host_data);
    std::vector<int64_t> shape;
    for (int i = 0; i < 4 && t->dims[i] != 0; ++i)
        shape.push_back(t->dims[i]);
    save_npy_f32(path, data, shape.data(), (int)shape.size());
    delete[] data;
}

} // namespace tensor_dump
