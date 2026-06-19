#include "core/resource.h"

#ifdef BACKEND_VULKAN

#include "backend/vulkan/vulkan_context.h"
#include <stdexcept>
#include <format>

VulkanMemoryResource::VulkanMemoryResource(int device_id)
    : device_id_(device_id),lock_memory_(false)
{
    auto& ctx = VulkanContext::get();
    device_ = ctx.device(device_id_);
}

VulkanMemoryResource::~VulkanMemoryResource() {
    if (buffer_) device_.destroyBuffer(buffer_);
    if (memory_) device_.freeMemory(memory_);
}

void* VulkanMemoryResource::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;

    auto& ctx = VulkanContext::get();

    vk::BufferCreateInfo buf_info(
        {}, size,
        vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::SharingMode::eExclusive);

    auto result = device_.createBuffer(&buf_info, nullptr, &buffer_);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format(
            "VulkanMemoryResource[dev={}]: createBuffer failed ({})",
            device_id_, vk::to_string(result)));
    }

    auto mem_reqs = device_.getBufferMemoryRequirements(buffer_);
    auto mem_props = ctx.physical_device(device_id_).getMemoryProperties();

    uint32_t mem_type_idx = UINT32_MAX;
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_reqs.memoryTypeBits & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            mem_type_idx = i;
            break;
        }
    }
    if (mem_type_idx == UINT32_MAX) {
        throw std::runtime_error("VulkanMemoryResource: no device-local memory type found");
    }

    vk::MemoryAllocateInfo alloc_info(mem_reqs.size, mem_type_idx);
    auto alloc_result = device_.allocateMemory(&alloc_info, nullptr, &memory_);
    if (alloc_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format(
            "VulkanMemoryResource[dev={}]: allocateMemory {} bytes failed ({})",
            device_id_, mem_reqs.size, vk::to_string(alloc_result)));
    }

    device_.bindBufferMemory(buffer_, memory_, 0);
    buffer_handle_ = reinterpret_cast<size_t>(static_cast<VkBuffer>(buffer_));
    ctx.registerBuffer(buffer_handle_, device_id_);

    return nullptr;
}

void VulkanMemoryResource::deallocate(void* ptr, size_t size) {
    // 资源在析构函数那里释放
}

#endif // BACKEND_VULKAN
