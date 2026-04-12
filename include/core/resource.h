#pragma once
#include "utils/utils.hpp"

struct MemoryBlock {
    void* ptr = nullptr;
    size_t size = 0;
    size_t offset = 0;
};

class IMemoryResource {
public:
    virtual ~IMemoryResource() = default;
    // 核心
    virtual void* allocate(size_t size, size_t alignment) = 0;
    virtual void deallocate(void* ptr, size_t size) = 0;

    virtual size_t id() const = 0;
    virtual Device device() const = 0;

    // virtual size_t allocated_size() const = 0; 
    // virtual size_t available_size() const = 0;
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
public:
    explicit CudaMemoryResource(int device_id) : device_id_(device_id) {}
    void* allocate(size_t size, size_t alignment) override;
    void deallocate(void* ptr, size_t size) override;
    Device device() const override { return Device::CUDA; }
    size_t id() const override { return static_cast<size_t>(device_id_); }
};
#endif