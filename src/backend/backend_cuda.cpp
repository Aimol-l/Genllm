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

    size_t free_memory = 0;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_memory, nullptr);
    size_t used_memory = total_memory - free_memory;

    double compute_power = 0.0;
    int sm = prop.major * 10 + prop.minor;

    // CUDA 12+ 弃用了 prop.clockRate，用 cudaDeviceGetAttribute 替代
    int clock_rate_khz = 0;
    cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device_id);
    double clock_ghz = static_cast<double>(clock_rate_khz) / 1e6;
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
    // CUDA 12+ 弃用了 prop.memoryClockRate，用 cudaDeviceGetAttribute 替代
    int mem_clock_khz = 0;
    int mem_bus_bits = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device_id);
    cudaDeviceGetAttribute(&mem_bus_bits, cudaDevAttrGlobalMemoryBusWidth, device_id);
    if (mem_bus_bits > 0) {
        bandwidth = 2.0 * mem_clock_khz * 1e3 * (mem_bus_bits / 8) / 1e9;
    }

    return BackendInfo{
        static_cast<size_t>(device_id),
        total_memory,
        used_memory,
        compute_power,
        bandwidth,
        Device::CUDA,
    };
}

void CUDABackendProvider::print_device_info(int device_id) const {
    auto info = get_backend_info(device_id);
    std::println("  Device Name:        {}", get_device_name(device_id));
    std::println("  Compute Capability: {}.{}", get_sm_version(device_id) / 10, get_sm_version(device_id) % 10);
    std::println("  Global Memory:      {:.1f} GB", static_cast<double>(info.total_memory) / (1ULL << 30));
    std::println("  Available Memory:   {:.1f} GB", static_cast<double>(info.available_memory()) / (1ULL << 30));
    std::println("  Max threads/block:  {}", get_max_threads_per_block(device_id));
    auto bd = get_max_block_dims(device_id);
    std::println("  Max block dims:     [{}, {}, {}]", bd[0], bd[1], bd[2]);
    auto gs = get_max_grid_size(device_id);
    std::println("  Max grid size:      [{}, {}, {}]", gs[0], gs[1], gs[2]);
}

static struct CUDABackendProviderRegistrar {
    CUDABackendProviderRegistrar() {
        BackendRegistry::instance().register_provider(std::make_unique<CUDABackendProvider>());
    }
} g_cuda_backend_registrar;
