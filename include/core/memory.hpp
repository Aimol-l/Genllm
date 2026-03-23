// memory.hpp - 内存管理
#pragma once
#include <memory>
#include <vector>
#include <cstdint>
#include <print>
#include <cstddef>
#include <mutex>
#include "utils/utils.hpp"

// ==================== 内存属性 ====================

enum class MemoryProperty {
    DEVICE_LOCAL,      // 设备本地内存（GPU显存）
    HOST_VISIBLE,      // 主机可见内存
    HOST_COHERENT,     // 主机一致内存
    DEVICE_UNCACHED    // 设备非缓存内存
};

// ==================== 内存句柄 ====================

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

// ==================== 内存资源接口 ====================

class IMemoryResource {
public:
    virtual ~IMemoryResource() = default;
    virtual void free(void* ptr) = 0;
    virtual int get_device_id() const = 0;
    virtual Backend get_backend() const = 0;
    virtual MemoryProperty get_property() const = 0;
    virtual void* allocate(size_t size, size_t alignment) = 0;
};

// ==================== CPU 内存资源 ====================

class CpuMemoryResource : public IMemoryResource {
public:
    CpuMemoryResource() = default;
    ~CpuMemoryResource() override = default;

    void free(void* ptr) override {
        if (ptr) {
            std::free(ptr);
        }
    }

    int get_device_id() const override { return 0; }
    Backend get_backend() const override { return Backend::CPU; }
    MemoryProperty get_property() const override { return MemoryProperty::HOST_VISIBLE; }

    void* allocate(size_t size, size_t alignment) override {
        if (alignment < 8) alignment = 8;
#ifdef _WIN32
        return _aligned_malloc(size, alignment);
#else
        void* ptr = nullptr;
        posix_memalign(&ptr, alignment, size);
        return ptr;
#endif
    }
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
    IMemoryResource* m_resource;  // 不负责生命周期

    std::vector<void*> m_allocated_chunks;

public:
    MemoryPool(IMemoryResource* resource, size_t chunk_size = 64ULL << 20)  // 默认 64MB
        : m_total_size(0)
        , m_used_size(0)
        , m_chunk_size(chunk_size)
        , m_resource(resource) {}

    virtual ~MemoryPool() {
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
    virtual MemoryHandle allocate(size_t size, size_t alignment = 8) {
        std::lock_guard<std::mutex> lock(m_mutex);

        // 简化实现：每次都直接分配
        // TODO: 实现真正的内存池分配策略（复用内存块）

        void* ptr = m_resource->allocate(size, alignment);
        if (!ptr) {
            throw std::runtime_error("Memory allocation failed");
        }

        m_allocated_chunks.push_back(ptr);
        m_used_size += size;

        return MemoryHandle(ptr, ptr, 0, size, alignment);
    }

    // 释放内存
    virtual void free(const MemoryHandle& handle) {
        if (!handle.is_valid) return;

        std::lock_guard<std::mutex> lock(m_mutex);

        // 简化实现：不立即释放，等待 trim() 时统一释放
        // TODO: 实现内存复用

        m_used_size -= handle.size;
    }

    // 整理内存池，释放未使用的内存
    virtual void trim() {
        std::lock_guard<std::mutex> lock(m_mutex);
        // TODO: 实现真正的 trim 逻辑
    }

    [[nodiscard]] virtual size_t get_total_allocated() const {
        return m_used_size;
    }

    [[nodiscard]] virtual size_t get_total_size() const {
        return m_total_size;
    }

    [[nodiscard]] virtual double get_usage_ratio() const {
        return m_total_size > 0 ? static_cast<double>(m_used_size) / m_total_size : 0.0;
    }

    [[nodiscard]] IMemoryResource* get_resource() const {
        return m_resource;
    }

    [[nodiscard]] Backend get_backend() const {
        return m_resource ? m_resource->get_backend() : Backend::CPU;
    }

    [[nodiscard]] int get_device_id() const {
        return m_resource ? m_resource->get_device_id() : 0;
    }
};