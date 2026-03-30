// memory.hpp - 内存管理
#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include <print>
#include <cstddef>
#include <mutex>
#include "utils/utils.hpp"

enum class MemoryProperty {
    DEVICE_LOCAL,      // 设备本地内存（GPU显存）
    HOST_VISIBLE,      // 主机可见内存
    HOST_COHERENT,     // 主机一致内存
    DEVICE_UNCACHED    // 设备非缓存内存
};
struct MemoryHandle {
    void* base_ptr;          // 基地址（内存池的基地址）
    void* data_ptr;          // 数据指针（分配的实际地址）
    size_t offset;           // 从基地址的偏移（字节）
    size_t size;             // 分配大小（字节）
    size_t alignment;        // 对齐要求
    bool is_valid;           // 是否有效
    MemoryHandle()
        : base_ptr(nullptr)
        , data_ptr(nullptr)
        , offset(0)
        , size(0)
        , alignment(8)
        , is_valid(false) {}

    MemoryHandle(void* base, void* data, size_t off, size_t sz, size_t align = 8)
        : base_ptr(base)
        , data_ptr(data)
        , offset(off)
        , size(sz)
        , alignment(align)
        , is_valid(true) {}

    void reset() {
        base_ptr = nullptr;
        data_ptr = nullptr;
        offset = 0;
        size = 0;
        alignment = 8;
        is_valid = false;
    }

    explicit operator bool() const {
        return is_valid;
    }
};

class IMemoryResource {
public:
    virtual ~IMemoryResource() = default;
    virtual void free(void* ptr) = 0;
    virtual int get_device_id() const = 0;
    virtual Device get_backend() const = 0;
    virtual MemoryProperty get_property() const = 0;
    virtual void* allocate(size_t size, size_t alignment) = 0;
};
// ==================== CPU 内存资源 ====================
class CpuMemoryResource : public IMemoryResource {
public:
    CpuMemoryResource() = default;
    ~CpuMemoryResource() override = default;

    void free(void* ptr) override;
    int get_device_id() const override { return 0; }
    void* allocate(size_t size, size_t alignment) override;
    Device get_backend() const override { return Device::CPU; }
    MemoryProperty get_property() const override { return MemoryProperty::HOST_VISIBLE; }
};
// ==================== CUDA 内存资源 ====================
#if GENLLM_HAS_CUDA

class CudaMemoryResource : public IMemoryResource {
private:
    int m_device_id;

public:
    explicit CudaMemoryResource(int device_id) : m_device_id(device_id) {
        cudaSetDevice(device_id);
    }

    ~CudaMemoryResource() override = default;

    void free(void* ptr) override {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    int get_device_id() const override { return m_device_id; }
    Backend get_backend() const override { return Backend::CUDA; }
    MemoryProperty get_property() const override { return MemoryProperty::DEVICE_LOCAL; }

    void* allocate(size_t size, size_t alignment) override {
        void* ptr = nullptr;
        cudaSetDevice(m_device_id);
        cudaMalloc(&ptr, size);
        return ptr;
    }
};

#endif


// ==================== 内存池 ====================
class MemoryPool {
protected:
    std::mutex m_mutex;
    size_t m_total_size;
    size_t m_used_size;
    size_t m_chunk_size;
    std::unique_ptr<IMemoryResource> m_weights_pool;
    std::unique_ptr<IMemoryResource> m_activate_pool;
    // std::unique_ptr<IMemoryResource> m_kv_cache_pool;

    std::vector<void*> m_allocated_chunks;
public:
    MemoryPool(IMemoryResource* resource, size_t chunk_size = 64ULL << 20)  // 默认 64MB
        : m_total_size(0)
        , m_used_size(0)
        , m_chunk_size(chunk_size)
        , m_resource(resource) {}

     ~MemoryPool() {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (void* chunk : m_allocated_chunks) {
            if (chunk) {
                m_resource->free(chunk);
            }
        }
        m_allocated_chunks.clear();
    }

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    // 分配内存
    MemoryHandle allocate(size_t size, size_t alignment = 8);

    // 释放内存
     void free(const MemoryHandle& handle);

    // 整理内存池，释放未使用的内存
    void trim() {
        std::lock_guard<std::mutex> lock(m_mutex);
        // TODO: 实现真正的 trim 逻辑
    }

    [[nodiscard]]  size_t get_total_allocated() const {
        return m_used_size;
    }

    [[nodiscard]]  size_t get_total_size() const {
        return m_total_size;
    }

    [[nodiscard]]  double get_usage_ratio() const {
        return m_total_size > 0 ? static_cast<double>(m_used_size) / m_total_size : 0.0;
    }
    [[nodiscard]] Device get_backend() const {
        return m_resource ? m_resource->get_backend() : Device::CPU;
    }

    [[nodiscard]] int get_device_id() const {
        return m_resource ? m_resource->get_device_id() : 0;
    }
};