#include "core/pools.h"

MemoryPool::MemoryPool(
    std::unique_ptr<IMemoryResource> resource,
    size_t capacity,
    std::string name)
    : resource_(std::move(resource))
    , capacity_(capacity)
    , name_(std::move(name))
{
    if (this->capacity_ > 0) {
        this->buffer_ = resource_->allocate(this->capacity_, 256);
    }
}

MemoryBlock MemoryPool::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return {nullptr, 0, 0};
    }
    if (alignment == 0) alignment = 1;

    size_t aligned_cursor = (this->cursor_ + alignment - 1) & ~(alignment - 1);

    if (aligned_cursor + size > this->capacity_) {
        throw std::runtime_error(std::format(
            "MemoryPool '{}': out of memory, need {} bytes, remaining {} bytes (peak={:.1f} MB)",
            name_, size, this->capacity_ - aligned_cursor,
            static_cast<double>(peak_) / (1ULL << 20)));
    }

    MemoryBlock block;
    block.ptr = static_cast<char*>(this->buffer_) + aligned_cursor;
    block.size = size;
    block.offset = aligned_cursor;


    this->cursor_ = aligned_cursor + size;
    this->used_ = aligned_cursor + size;

    if (this->used_ > peak_) 
        peak_ = this->used_;

    return block;
}

void MemoryPool::reset() {
    this->cursor_ = 0;
    this->used_ = 0;
}

std::string MemoryPool::format_usage() const {
    return std::format("{}:{} used={:.1f}/{:.1f} MB peak={:.1f} MB ({:.1f}%)",
        device_to_string(device()), device_id(),
        static_cast<double>(this->used_) / (1ULL << 20),
        static_cast<double>(this->capacity_) / (1ULL << 20),
        static_cast<double>(peak_) / (1ULL << 20),
        utilization() * 100.0);
}
