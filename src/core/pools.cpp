#include "core/pools.h"

MemoryPool::MemoryPool(std::unique_ptr<IMemoryResource> resource,
                       size_t capacity,
                       std::string name)
    : resource_(std::move(resource))
    , capacity_(capacity)
    , name_(std::move(name))
{
    if (capacity_ > 0) {
        buffer_ = resource_->allocate(capacity_, 256);
    }
    std::println("[mem] created pool '{}' on {}:{} capacity={:.1f} MB",
                 name_, device_to_string(device()), device_id(),
                 static_cast<double>(capacity_) / (1ULL << 20));
}

MemoryPool::~MemoryPool() {
    if (buffer_ && capacity_ > 0) {
        resource_->deallocate(buffer_, capacity_);
    }
}

MemoryBlock MemoryPool::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return {nullptr, 0, 0};
    }

    size_t aligned_cursor = (cursor_ + alignment - 1) & ~(alignment - 1);
    if (aligned_cursor + size > capacity_) {
        throw std::runtime_error(std::format(
            "MemoryPool '{}': out of memory, need {} bytes, remaining {} bytes (peak={:.1f} MB)",
            name_, size, capacity_ - aligned_cursor,
            static_cast<double>(peak_) / (1ULL << 20)));
    }

    MemoryBlock block;
    block.ptr = static_cast<char*>(buffer_) + aligned_cursor;
    block.size = size;
    block.offset = aligned_cursor;

    cursor_ = aligned_cursor + size;
    used_ = cursor_;
    if (used_ > peak_) peak_ = used_;

    return block;
}

void MemoryPool::reset() {
    cursor_ = 0;
    used_ = 0;
}

std::string MemoryPool::format_usage() const {
    return std::format("{}:{} used={:.1f}/{:.1f} MB peak={:.1f} MB ({:.1f}%)",
        device_to_string(device()), device_id(),
        static_cast<double>(used_) / (1ULL << 20),
        static_cast<double>(capacity_) / (1ULL << 20),
        static_cast<double>(peak_) / (1ULL << 20),
        utilization() * 100.0);
}
