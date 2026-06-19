#pragma once
#include <memory>
#include <mutex>
#ifndef __CUDACC__
#include <print>
#endif
#include <iostream>
#include <format>
#include "resource.h"

class MemoryPool {
private:
    void* buffer_ = nullptr;
    size_t capacity_;
    size_t used_ = 0;
    size_t peak_ = 0;
    size_t cursor_ = 0;
    std::string name_;
    mutable std::mutex mutex_;

    std::unique_ptr<IMemoryResource> resource_;
public:
    void reset();
    void reset_to(size_t pos);

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
    [[nodiscard]] size_t device_handle() const { return resource_->device_handle(); }
    [[nodiscard]] const std::string& name() const { return name_; }
};

struct DevicePools {
    std::unique_ptr<MemoryPool> weight;
    std::unique_ptr<MemoryPool> activation;
    std::unique_ptr<MemoryPool> kv_cache;
    void reset_activation() {
        if (activation) {
            std::cout<<std::format("Reset activation pool on dev {}",activation->format_usage())<<std::endl;
            activation->reset();
        }
    }
    void print_usage() const {
        if (weight) 
            std::cout<<std::format("  weight:     {}", weight->format_usage())<<std::endl;
        if (activation) 
            std::cout<<std::format("  activation: {}", activation->format_usage())<<std::endl;
        if (kv_cache) 
            std::cout<<std::format("  kv_cache:   {}", kv_cache->format_usage())<<std::endl;
    }
};