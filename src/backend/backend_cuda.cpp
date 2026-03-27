#if defined(USE_CUDA)
#include "backend/backend.h"

// ==================== CUDA 后端实现 ====================

bool CUDABackendProvider::is_available() const {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

int CUDABackendProvider::get_device_count() const {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

BackendInfo CUDABackendProvider::get_device_info(int device_id) const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return BackendInfo(Backend::CUDA, static_cast<size_t>(device_id),
                       prop.totalGlobalMem, prop.name);
}

#endif
