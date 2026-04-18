#pragma once
#include <memory>
#include <print>
#include "resource.h"

class MemoryPool {
private:
    void* buffer_ = nullptr;
    size_t capacity_;
    size_t used_ = 0;
    size_t peak_ = 0;
    size_t cursor_ = 0;
    std::string name_;
    std::unique_ptr<IMemoryResource> resource_;
public:
    void reset();
    void reset_to(size_t pos);  // 回退到指定位置（保留 pos 之前的数据）
    explicit MemoryPool(std::unique_ptr<IMemoryResource> resource,size_t capacity,std::string name);
    ~MemoryPool(){
        if (buffer_ && capacity_ > 0) {
            resource_->deallocate(buffer_, capacity_);
        }
    }
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    MemoryBlock allocate(size_t size, size_t alignment);

    std::string format_usage() const;
    [[nodiscard]] size_t capacity() const { return capacity_; }
    [[nodiscard]] size_t used() const { return used_; }
    [[nodiscard]] size_t peak() const { return peak_; }
    [[nodiscard]] double utilization() const {
        return capacity_ > 0 ? static_cast<double>(used_) / capacity_ : 0.0;
    }
    [[nodiscard]] Device device() const { return resource_->device(); }
    [[nodiscard]] size_t device_id() const { return resource_->id(); }
    [[nodiscard]] const std::string& name() const { return name_; }
};

struct DevicePools {
    std::unique_ptr<MemoryPool> weight;
    std::unique_ptr<MemoryPool> activation;
    std::unique_ptr<MemoryPool> kv_cache;
    void reset_activation() {
        if (activation) {
            std::println("Reset activation pool on dev {}",activation->format_usage());
            activation->reset();
        }
    }
    void print_usage() const {
        if (weight) std::println("  weight:     {}", weight->format_usage());
        if (activation) std::println("  activation: {}", activation->format_usage());
        if (kv_cache) std::println("  kv_cache:   {}", kv_cache->format_usage());
    }
};