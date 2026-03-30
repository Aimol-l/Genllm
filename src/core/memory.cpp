#include "core/memory.hpp"

MemoryHandle  MemoryPool::allocate(size_t size, size_t alignment = 8) {
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

void MemoryPool::free(const MemoryHandle& handle) {
    if (!handle.is_valid) return;

    std::lock_guard<std::mutex> lock(m_mutex);

    // 简化实现：不立即释放，等待 trim() 时统一释放
    // TODO: 实现内存复用

    m_used_size -= handle.size;
}
void MemoryPool::trim() {
    std::lock_guard<std::mutex> lock(m_mutex);
    // TODO: 实现真正的 trim 逻辑
}

// ==================== CPU 内存资源 ====================
void* CpuMemoryResource::allocate(size_t size, size_t alignment){
    if (alignment < 8) alignment = 8;
    #ifdef _WIN32
    return _aligned_malloc(size, alignment);
    #else
    void* ptr = nullptr;
    posix_memalign(&ptr, alignment, size);
    return ptr;
    #endif
}
void CpuMemoryResource::free(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}