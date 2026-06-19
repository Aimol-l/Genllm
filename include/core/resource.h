#pragma once
#include "utils/utils.hpp"
#ifdef BACKEND_CUDA
#include <driver_types.h>
#endif

struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    size_t offset = 0;
    size_t device_handle = 0;   // Vulkan: VkBuffer handle；CPU/CUDA: 0
};

class IMemoryResource {
public:
    virtual ~IMemoryResource() = default;
    // 核心
    virtual void deallocate(void* ptr, size_t size) = 0;
    virtual void* allocate(size_t size, size_t alignment) = 0;

    virtual size_t id() const = 0;
    virtual Device device() const = 0;
    virtual size_t device_handle() const { return 0; }  // Vulkan: VkBuffer handle；CPU/CUDA: 0
};

class CpuMemoryResource : public IMemoryResource {
    bool lock_memory_ = false;
public:
    explicit CpuMemoryResource(bool lock_memory = false) : lock_memory_(lock_memory) {}

    void* allocate(size_t size, size_t alignment) override;
    void deallocate(void* ptr, size_t size) override;
    Device device() const override { return Device::CPU; }
    size_t id() const override { return 0; }
    [[nodiscard]] bool lock_memory() const { return lock_memory_; }
};

#ifdef BACKEND_CUDA
class CudaMemoryResource : public IMemoryResource {
    int device_id_;
    bool lock_memory_ = false;
    cudaStream_t stream_;
public:
    explicit CudaMemoryResource(int device_id) : device_id_(device_id) {}
    void deallocate(void* ptr, size_t size) override;
    void* allocate(size_t size, size_t alignment) override;
    [[nodiscard]] bool lock_memory() const { return lock_memory_; }
    [[nodiscard]] Device device() const override { return Device::CUDA; }
    [[nodiscard]] size_t id() const override { return static_cast<size_t>(device_id_); }
};
#endif

#ifdef BACKEND_VULKAN
#include <vulkan/vulkan.hpp>

class VulkanMemoryResource : public IMemoryResource {
    int device_id_;
    size_t buffer_handle_;
    bool lock_memory_ = false;
    vk::Device device_;
    vk::Buffer buffer_;
    vk::DeviceMemory memory_;
public:
    explicit VulkanMemoryResource(int device_id);
    ~VulkanMemoryResource() override;
    void* allocate(size_t size, size_t alignment) override;
    void deallocate(void* ptr, size_t size) override;

    [[nodiscard]] bool lock_memory() const { return lock_memory_; }
    [[nodiscard]] Device device() const override { return Device::VULKAN; }
    [[nodiscard]] size_t id() const override { return static_cast<size_t>(device_id_); }
    [[nodiscard]] size_t device_handle() const override { return buffer_handle_; }
};
#endif