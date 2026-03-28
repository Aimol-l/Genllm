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
const BackendInfo* DeviceManager::get_device(Backend backend, size_t device_id) {
    for (const auto& dev : get_devices()) {
        if (dev.backend == backend && dev.device_id == device_id) {
            return &dev;
        }
    }
    return nullptr;
}

void DeviceManager::print_devices() {
    // 打印已注册后端
    auto& registry = BackendRegistry::instance();
    std::println("Registered backends: {}", registry.get_providers().size());
    for (const auto& provider : registry.get_providers()) {
        std::println("  {}: {} device(s)",
            provider->get_backend_name(),
            provider->get_device_count());
    }
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
