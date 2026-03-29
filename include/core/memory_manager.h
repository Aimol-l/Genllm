#pragma once
#include <unordered_map>
#include <memory>
#include <vector>
#include <print>
#include <format>
#include "memory.hpp"
#include "backend/backend.h"
#include "tensor.hpp"

// 内存管理器：为每个设备管理独立的内存池
class MemoryManager {
private:
    // eg： cuda --> pool(weights,activate,kv-cache)
    std::unordered_map<std::string, std::unique_ptr<MemoryPool>> m_pools;
    std::unordered_map<std::string, std::unique_ptr<IMemoryResource>> m_resources;

    // 为设备创建内存资源
    IMemoryResource* create_memory_resource(const BackendInfo& device) {
        std::string key = get_device_key(device);

        if (m_resources.find(key) != m_resources.end()) {
            return m_resources[key].get();
        }

        IMemoryResource* resource = nullptr;

        switch (device.device) {
            case Device::CPU:
                resource = new CpuMemoryResource();
                break;

#if GENLLM_HAS_CUDA
            case Backend::CUDA:
                resource = new CudaMemoryResource(static_cast<int>(device.device_id));
                break;
#endif

#if GENLLM_HAS_VULKAN
            case Backend::Vulkan:
                // TODO: 实现 Vulkan 内存资源
                throw std::runtime_error("Vulkan memory resource not implemented");
                break;
#endif

            default:
                throw std::runtime_error("Unsupported backend");
        }

        m_resources[key] = std::unique_ptr<IMemoryResource>(resource);
        return resource;
    }

    // 生成设备唯一键
    static std::string get_device_key(const BackendInfo& device) {
        return std::format("{}:{}", static_cast<int>(device.device), device.id);
    }

public:
    MemoryManager() = default;
    ~MemoryManager() = default;

    // 禁止拷贝
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // 获取或创建设备的内存池
    MemoryPool* get_or_create_pool(const BackendInfo& device, size_t chunk_size = 64ULL << 20) {
        std::string key = get_device_key(device);

        if (m_pools.find(key) == m_pools.end()) {
            IMemoryResource* resource = create_memory_resource(device);
            m_pools[key] = std::make_unique<MemoryPool>(resource, chunk_size);
            std::println("Created memory pool for {} (chunk_size={} MB)",device.to_string(), chunk_size / (1024 * 1024));
        }

        return m_pools[key].get();
    }

    // 获取设备的内存池
    [[nodiscard]] MemoryPool* get_pool(const BackendInfo& device) const {
        std::string key = get_device_key(device);
        auto it = m_pools.find(key);
        return it != m_pools.end() ? it->second.get() : nullptr;
    }

    // ==================== 张量内存分配 ====================

    // 为张量分配内存
    MemoryHandle allocate_tensor(Tensor* tensor, const BackendInfo& device) {
        if (!tensor) {
            throw std::runtime_error("Cannot allocate memory for null tensor");
        }

        // 计算张量大小
        size_t num_elements = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && tensor->dims[i] != 0; ++i) {
            num_elements *= tensor->dims[i];
        }

        size_t size = num_elements * data_type_size(tensor->dtype);

        // 从内存池分配
        MemoryPool* pool = get_or_create_pool(device);
        MemoryHandle handle = pool->allocate(size, 8);  // 默认8字节对齐

        // 设置张量的内存信息
        tensor->data = handle.data_ptr;
        tensor->offset = handle.offset;
        tensor->backend = pool->get_resource();
        // TODO: 设置 device_id

        return handle;
    }

    // 为权重张量分配内存
    MemoryHandle allocate_weight(Tensor* weight, const BackendInfo& device) {
        std::println("Allocating weight {} on {} ({} MB)",
            weight->name,
            device.to_string(),
            get_tensor_size_mb(weight));

        return allocate_tensor(weight, device);
    }

    // 为临时张量分配内存
    MemoryHandle allocate_temp(Tensor* tensor, const BackendInfo& device) {
        return allocate_tensor(tensor, device);
    }

    // ==================== 工具函数 ====================

    // 计算张量大小（MB）
    static double get_tensor_size_mb(const Tensor* tensor) {
        if (!tensor) return 0.0;

        size_t num_elements = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && tensor->dims[i] != 0; ++i) {
            num_elements *= tensor->dims[i];
        }

        size_t size_bytes = num_elements * data_type_size(tensor->dtype);
        return static_cast<double>(size_bytes) / (1024.0 * 1024.0);
    }

    // 计算张量大小（字节）
    static size_t get_tensor_size_bytes(const Tensor* tensor) {
        if (!tensor) return 0;

        size_t num_elements = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && tensor->dims[i] != 0; ++i) {
            num_elements *= tensor->dims[i];
        }

        return num_elements * data_type_size(tensor->dtype);
    }

    // ==================== 统计信息 ====================

    // 打印内存统计
    void print_stats() const {
        std::println("\n=== Memory Pool Statistics ===");

        size_t total_allocated = 0;
        for (const auto& [key, pool] : m_pools) {
            size_t allocated = pool->get_total_allocated();
            total_allocated += allocated;
            std::println("  {}: {} MB allocated",
                key, allocated / (1024 * 1024));
        }

        std::println("Total: {} MB allocated", total_allocated / (1024 * 1024));
    }

    // 获取总内存使用量
    [[nodiscard]] size_t get_total_memory_used() const {
        size_t total = 0;
        for (const auto& [key, pool] : m_pools) {
            total += pool->get_total_allocated();
        }
        return total;
    }

    // 清理所有内存池
    void clear() {
        m_pools.clear();
        m_resources.clear();
    }
};
