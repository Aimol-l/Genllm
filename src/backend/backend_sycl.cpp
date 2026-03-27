#if defined(USE_SYCL)
#include "backend/backend.h"

// ==================== Sycl 后端实现 ====================

bool SyclBackendProvider::is_available() const {
    try {
        sycl::device device = sycl::device(sycl::default_selector_v);
        return device.is_gpu() || device.is_cpu() || device.is_accelerator();
    } catch (...) {
        return false;
    }
}

int SyclBackendProvider::get_device_count() const {
    try {
        std::vector<sycl::device> devices = sycl::device::get_devices();
        return static_cast<int>(devices.size());
    } catch (...) {
        return 0;
    }
}

BackendInfo SyclBackendProvider::get_device_info(int device_id) const {
    try {
        std::vector<sycl::device> devices = sycl::device::get_devices();
        if (device_id >= 0 && device_id < static_cast<int>(devices.size())) {
            auto& device = devices[device_id];
            size_t global_mem = device.get_info<sycl::info::device::global_mem_size>();
            std::string name = device.get_info<sycl::info::device::name>();
            return BackendInfo(Backend::Sycl, static_cast<size_t>(device_id), global_mem, name);
        }
    } catch (...) {
        // 忽略异常
    }
    return BackendInfo(Backend::Sycl, static_cast<size_t>(device_id), 0, "Sycl Device");
}

#endif
