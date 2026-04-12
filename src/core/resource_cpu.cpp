#include <format>
#include <cstring>
#include <print>

#if defined(_WIN32)
    #include <malloc.h>
    #include <windows.h>
#else
    #include <cstdlib>
    #include <sys/mman.h>
#endif

#include "core/resource.h"


void* CpuMemoryResource::allocate(size_t size, size_t alignment = 8) {
    if (size == 0) return nullptr;

    void* ptr = nullptr;

#if defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
    if (!ptr) {
        throw std::runtime_error(std::format("CpuMemoryResource: _aligned_malloc {} bytes failed", size));
    }
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        throw std::runtime_error(std::format("CpuMemoryResource: posix_memalign {} bytes (align={}) failed", size, alignment));
    }
#endif

    if (lock_memory_) {
#if defined(_WIN32)
        if (!VirtualLock(ptr, size)) { 
            std::println("[warn] VirtualLock {} bytes failed, continuing without lock", size);
        }
#else
        if (mlock(ptr, size) != 0) {
            std::println("[warn] mlock {} bytes failed (try: ulimit -l unlimited), continuing without lock", size);
        }
#endif
    }
    std::memset(ptr, 0, size);
    return ptr;
}

void CpuMemoryResource::deallocate(void* ptr, size_t size) {
    if (!ptr) return;

    if (lock_memory_ && size > 0) {
    #if defined(_WIN32)
            VirtualUnlock(ptr, size);
    #else
            munlock(ptr, size);
    #endif
        }

    #if defined(_WIN32)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
    std::println("Deallocated {} bytes on CPU", size);
}
