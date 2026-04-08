#include "backend/backend.h"
#include <cuda_runtime.h>
#include <array>

bool CUDABackendProvider::is_available() const {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0;
}

int CUDABackendProvider::get_device_count() const {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

std::string CUDABackendProvider::get_device_name(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.name;
}

int CUDABackendProvider::get_sm_version(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.major * 10 + prop.minor;
}

int CUDABackendProvider::get_max_threads_per_block(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.maxThreadsPerBlock;
}

std::array<int, 3> CUDABackendProvider::get_max_block_dims(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return {prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]};
}

std::array<int, 3> CUDABackendProvider::get_max_grid_size(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return {prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]};
}

BackendInfo CUDABackendProvider::get_backend_info(int device_id) const {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    size_t total_memory = prop.totalGlobalMem;
    size_t reserved = static_cast<size_t>(total_memory * 0.05);

    double compute_power = 0.0;
    int sm = prop.major * 10 + prop.minor;
    double clock_ghz = static_cast<double>(prop.clockRate) / 1e6;
    size_t sm_count = prop.multiProcessorCount;
    if (sm >= 80) {
        compute_power = 2.0 * sm_count * clock_ghz;
    } else if (sm >= 70) {
        compute_power = 1.3 * sm_count * clock_ghz;
    } else if (sm >= 60) {
        compute_power = 1.0 * sm_count * clock_ghz;
    } else {
        compute_power = 0.5 * sm_count * clock_ghz;
    }

    double bandwidth = 0.0;
    if (prop.major >= 3) {
        bandwidth = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
    }

    return BackendInfo{
        static_cast<size_t>(device_id),
        Device::CUDA,
        total_memory,
        reserved,
        compute_power,
        bandwidth
    };
}

static struct CUDABackendProviderRegistrar {
    CUDABackendProviderRegistrar() {
        BackendRegistry::instance().register_provider(std::make_unique<CUDABackendProvider>());
    }
} g_cuda_backend_registrar;
