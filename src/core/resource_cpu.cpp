#include <print>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include "core/resource.h"

void* CpuMemoryResource::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        throw std::runtime_error(std::format(
            "CpuMemoryResource: failed to allocate {} bytes (align={})", size, alignment));
    }

    if (lock_memory_) {
        if (mlock(ptr, size) != 0) {
            std::println("[warn] mlock {} bytes failed (try: sudo sysctl -w vm.max_map_count=262144), continuing without lock", size);
        }
    }

    std::memset(ptr, 0, size);
    return ptr;
}

void CpuMemoryResource::deallocate(void* ptr, size_t size) {
    if (ptr && size > 0) {
        if (lock_memory_) munlock(ptr, size);
    }
    free(ptr);
}
