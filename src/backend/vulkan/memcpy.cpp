#include "backend/vulkan/memcpy.h"
#include "tensor.hpp"
#include "utils/utils.hpp"


#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>
#include "backend/vulkan/vulkan_context.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace ops {

void MemcpyImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    Tensor* src = out->src[0];   // 源
    Tensor* dst = out;           // 目标就是 out 自身
    if (!src || !dst) {
        throw std::runtime_error("MemcpyImpl<VULKAN>: src or dst is null");
    }

    size_t nbytes = out->bytes();  // 拷贝字节数
    Device src_dev = src->device;
    Device dst_dev = dst->device;  // 即 out->device

    auto& ctx = VulkanContext::get();

    // ── CPU → Vulkan ──
    if (src_dev == Device::CPU && dst_dev == Device::VULKAN) {
        VkBuffer dst_buffer = reinterpret_cast<VkBuffer>(dst->device_handle);
        VkDeviceSize dst_offset = dst->offset;
        // 创建 staging buffer
        vk::DeviceMemory staging_mem;
        void* staging_ptr = nullptr;
        vk::Buffer staging_buf = ctx.createStagingBuffer(dev_id, nbytes, &staging_mem, &staging_ptr);
        if (!staging_buf) throw std::runtime_error("MemcpyImpl: staging buffer creation failed");

        std::memcpy(staging_ptr, src->data, nbytes);  // CPU 数据拷贝

        if (!ctx.isRecording(dev_id)) throw std::runtime_error("MemcpyImpl: not in recording mode");
        
        vk::CommandBuffer cmd = ctx.cmdBuffer(dev_id);
        cmd.copyBuffer(staging_buf, dst_buffer, vk::BufferCopy(0, dst_offset, nbytes));

        // Barrier: 拷贝完成 → 后续 compute shader 可读
        ctx.addBufferBarrier(dev_id, dst_buffer,
            vk::AccessFlagBits::eTransferWrite,
            vk::AccessFlagBits::eShaderRead,
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eComputeShader);

        // 延迟释放 staging buffer
        auto& slot = ctx.slot(dev_id);
        std::lock_guard<std::mutex> lock(*slot.pending_frees_mutex);
        slot.pending_staging_frees.emplace_back(staging_buf, staging_mem, staging_ptr);
        return;
    }

    // ── Vulkan → CPU ──
    if (src_dev == Device::VULKAN && dst_dev == Device::CPU) {
        VkBuffer src_buffer = reinterpret_cast<VkBuffer>(src->device_handle);
        VkDeviceSize src_offset = src->offset;
        void* dst_cpu = dst->data;
        // 创建 staging buffer（作为拷贝目标，CPU 可读）
        vk::DeviceMemory staging_mem;
        void* staging_ptr = nullptr;
        vk::Buffer staging_buf = ctx.createStagingBuffer(dev_id, nbytes, &staging_mem, &staging_ptr);
        if (!staging_buf) throw std::runtime_error("MemcpyImpl: staging buffer creation failed");

        if (!ctx.isRecording(dev_id)) throw std::runtime_error("MemcpyImpl: not in recording mode");
        vk::CommandBuffer cmd = ctx.cmdBuffer(dev_id);

        // Barrier：确保之前所有 compute shader 写入完成后再执行拷贝
        ctx.addBufferBarrier(dev_id, src_buffer,
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eTransferRead,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer);

        // 录制拷贝命令：Vulkan buffer → staging buffer
        cmd.copyBuffer(src_buffer, staging_buf, vk::BufferCopy(src_offset, 0, nbytes));

        // 记录回读任务，在 cleanup 时执行 memcpy
        auto& slot = ctx.slot(dev_id);
        std::lock_guard<std::mutex> lock(*slot.pending_frees_mutex);
        slot.pending_readbacks.push_back({staging_buf, staging_mem, staging_ptr, dst_cpu, nbytes});
        return;
    }

    throw std::runtime_error("MemcpyImpl: unsupported device combination");
}

template struct MemcpyImpl<Device::VULKAN>;

}

#endif
