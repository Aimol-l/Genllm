#include <print>
#include "core/resource.h"

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif


#ifdef BACKEND_CUDA

void* CudaMemoryResource::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::format(
            "CudaMemoryResource[dev={}]: cudaMalloc {} bytes failed: {}",
            device_id_, size, cudaGetErrorString(err)));
    }
    return ptr;
}

void CudaMemoryResource::deallocate(void* ptr, size_t size) {
    if (ptr) cudaFree(ptr);
    // std::println("Deallocated {} bytes on CUDA{}", size, device_id_);
}

#endif