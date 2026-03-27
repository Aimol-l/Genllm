#include "backend/backend.h"

// ==================== BackendRegistry 实现 ====================

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    return registry;
}

void BackendRegistry::register_provider(std::unique_ptr<BackendProvider> provider) {
    if (provider && provider->is_available()) {
        providers_.push_back(std::move(provider));
    }
}

// ==================== DeviceManager 实现 ====================

DeviceManager& DeviceManager::instance() {
    static DeviceManager manager;
    return manager;
}

void DeviceManager::detect_devices() {
    devices_.clear();
    auto& registry = BackendRegistry::instance();
    for (const auto& provider : registry.get_providers()) {
        int count = provider->get_device_count();
        for (int i = 0; i < count; ++i) {
            auto info = provider->get_device_info(i);
            devices_.emplace_back(info);
        }
    }
    initialized_ = true;
}

const std::vector<BackendInfo>& DeviceManager::get_devices() {
    if (!initialized_) {
        detect_devices();
    }
    return devices_;
}

size_t DeviceManager::device_count() {
    return get_devices().size();
}

std::vector<const BackendInfo*> DeviceManager::get_devices_by_backend(Backend backend) {
    std::vector<const BackendInfo*> result;
    for (const auto& dev : get_devices()) {
        if (dev.backend == backend) {
            result.push_back(&dev);
        }
    }
    return result;
}

const BackendInfo* DeviceManager::get_cpu_device() {
    for (const auto& dev : get_devices()) {
        if (dev.backend == Backend::CPU) {
            return &dev;
        }
    }
    return nullptr;
}

const BackendInfo* DeviceManager::get_device(Backend backend, size_t device_id) {
    for (const auto& dev : get_devices()) {
        if (dev.backend == backend && dev.device_id == device_id) {
            return &dev;
        }
    }
    return nullptr;
}

bool DeviceManager::has_backend(Backend backend) {
    return !get_devices_by_backend(backend).empty();
}

bool DeviceManager::has_cuda() {
    return has_backend(Backend::CUDA);
}

int DeviceManager::get_cuda_device_count() {
    return static_cast<int>(get_devices_by_backend(Backend::CUDA).size());
}

void DeviceManager::print_devices() {
    std::println("\n=== Available Devices ({}) ===", device_count());
    for (const auto& dev : get_devices()) {
        std::println("  {}", dev.to_string());
    }

    // 打印已注册后端
    auto& registry = BackendRegistry::instance();
    std::println("\nRegistered backends: {}", registry.get_providers().size());
    for (const auto& provider : registry.get_providers()) {
        std::println("  {}: {} device(s)",
            provider->get_backend_name(),
            provider->get_device_count());
    }
}

std::string DeviceManager::get_summary() {
    if (!initialized_) {
        detect_devices();
    }

    std::string summary = "Devices: ";
    bool first = true;
    for (const auto& dev : devices_) {
        if (!first) summary += ", ";
        summary += std::format("{}[{}]({}MB)",
            backend_to_string(dev.backend),
            dev.device_id,
            dev.total_memory / (1024 * 1024));
        first = false;
    }
    return summary;
}

// ==================== 后端初始化 ====================

namespace backend {

void init_builtin_backends() {
    // 注册所有内置后端
    REGISTER_BACKEND(CPUBackendProvider);

#if defined(USE_CUDA)
    REGISTER_BACKEND(CUDABackendProvider);
#endif

#if defined(USE_VULKAN)
    REGISTER_BACKEND(VulkanBackendProvider);
#endif

#if defined(USE_SYCL)
    REGISTER_BACKEND(SyclBackendProvider);
#endif
}

} // namespace backend
