// backend.h - 插件式后端设备管理
#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <print>
#include "utils/utils.hpp"

// ==================== 后端提供者接口 ====================
// 后端设备信息
struct BackendDeviceInfo {
    Backend type;
    size_t device_id;
    size_t total_memory;
    std::string name;
};

// 后端提供者：插件式接口
class BackendProvider {
public:
    virtual ~BackendProvider() = default;
    [[nodiscard]] virtual bool is_available() const = 0;
    [[nodiscard]] virtual int get_device_count() const = 0;
    [[nodiscard]] virtual Backend get_backend_type() const = 0;
    [[nodiscard]] virtual const char* get_backend_name() const = 0;
    [[nodiscard]] virtual BackendDeviceInfo get_device_info(int device_id) const = 0;
};

// ==================== 后端注册系统 ====================
class BackendRegistry {
private:
    BackendRegistry() = default;
    std::vector<std::unique_ptr<BackendProvider>> providers_;
public:
    // 获取单例
    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }
    // 注册后端提供者
    void register_provider(std::unique_ptr<BackendProvider> provider) {
        if (provider && provider->is_available()) {
            providers_.push_back(std::move(provider));
        }
    }
    // 获取所有已注册的提供者
    [[nodiscard]] const std::vector<std::unique_ptr<BackendProvider>>& get_providers() const {
        return providers_;
    }
    // 禁止拷贝和移动
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;
};

// ==================== 内置后端实现 ====================
// CPU 后端（总是可用）
class CPUBackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override { return true; }
    [[nodiscard]] int get_device_count() const override { return 1; }
    [[nodiscard]] Backend get_backend_type() const override { return Backend::CPU; }
    [[nodiscard]] const char* get_backend_name() const override { return "CPU"; }
    [[nodiscard]] BackendDeviceInfo get_device_info(int device_id) const override {
        size_t memory = get_system_memory();
        return {Backend::CPU, 0, memory, "CPU"};
    }
private:
    [[nodiscard]] static size_t get_system_memory() {
    #ifdef _WIN32
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return status.ullTotalPhys;
    #elif __linux__ || __APPLE__
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        return pages * page_size;
    #else
        return 8ULL << 30;  // 默认 8GB
    #endif
    }
};

// CUDA 后端
#if defined(USE_CUDA)
#include <cuda_runtime.h>

class CUDABackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }
    [[nodiscard]] int get_device_count() const override {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }
    [[nodiscard]] BackendDeviceInfo get_device_info(int device_id) const override {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        return {Backend::CUDA, static_cast<size_t>(device_id),prop.totalGlobalMem, prop.name};
    }
    [[nodiscard]] Backend get_backend_type() const override { return Backend::CUDA; }
    [[nodiscard]] const char* get_backend_name() const override { return "CUDA"; }
};
#endif

// Vulkan 后端
#if defined(USE_VULKAN)
class VulkanBackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override {
        // TODO: 实现 Vulkan 可用性检测
        return false;
    }
    [[nodiscard]] int get_device_count() const override { return 0; }
    [[nodiscard]] BackendDeviceInfo get_device_info(int device_id) const override {
        return {Backend::Vulkan, 0, 0, "Vulkan"};
    }
    [[nodiscard]] Backend get_backend_type() const override { return Backend::Vulkan; }
    [[nodiscard]] const char* get_backend_name() const override { return "Vulkan"; }
};
#endif

// Sycl 后端
#if defined(USE_SYCL)
class SyclBackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override {
        // TODO: 实现 Sycl 可用性检测
        return false;
    }
    [[nodiscard]] int get_device_count() const override { return 0; }
    [[nodiscard]] BackendDeviceInfo get_device_info(int device_id) const override {
        return {Backend::Sycl, 0, 0, "Sycl"};
    }
    [[nodiscard]] Backend get_backend_type() const override { return Backend::Sycl; }
    [[nodiscard]] const char* get_backend_name() const override { return "Sycl"; }
};
#endif

// ==================== 设备管理器 ====================

class DeviceManager {
private:
    bool initialized_ = false;
    std::vector<BackendDeviceInfo> devices_;
    DeviceManager() = default;
    void detect_devices() {
        devices_.clear();
        auto& registry = BackendRegistry::instance();
        for (const auto& provider : registry.get_providers()) {
            int count = provider->get_device_count();
            for (int i = 0; i < count; ++i) {
                auto info = provider->get_device_info(i);
                devices_.emplace_back(info.type, info.device_id,
                                      info.total_memory, info.name);
            }
        }
        initialized_ = true;
    }
public:
    // 获取单例
    static DeviceManager& instance() {
        static DeviceManager manager;
        return manager;
    }
    // 禁止拷贝和移动
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    // ==================== 设备查询 ====================
    // 获取所有设备（延迟初始化）
    [[nodiscard]] const std::vector<BackendDeviceInfo>& get_devices() {
        if (!initialized_) {
            detect_devices();
        }
        return devices_;
    }
    [[nodiscard]] size_t device_count() {
        return get_devices().size();
    }
    [[nodiscard]] std::vector<const BackendDeviceInfo*> get_devices_by_backend(Backend backend) {
        std::vector<const BackendDeviceInfo*> result;
        for (const auto& dev : get_devices()) {
            if (dev.backend == backend) {
                result.push_back(&dev);
            }
        }
        return result;
    }
    [[nodiscard]] const Device* get_cpu_device() {
        for (const auto& dev : get_devices()) {
            if (dev.backend == Backend::CPU) {
                return &dev;
            }
        }
        return nullptr;
    }
    [[nodiscard]] bool has_backend(Backend backend) {
        return !get_devices_by_backend(backend).empty();
    }
    void print_devices() {
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
};

// ==================== 便利函数 ====================

inline DeviceManager& device_manager() {
    return DeviceManager::instance();
}

inline const std::vector<Device>& get_available_devices() {
    return device_manager().get_devices();
}

inline void print_available_devices() {
    device_manager().print_devices();
}

// ==================== 自动注册宏 ====================

#define REGISTER_BACKEND(ProviderClass) \
    namespace { \
        struct ProviderClass##Registrar { \
            ProviderClass##Registrar() { \
                BackendRegistry::instance().register_provider( \
                    std::make_unique<ProviderClass>()); \
            } \
        }; \
        static ProviderClass##Registrar g_##ProviderClass##_registrar; \
    }

// ==================== 自动注册内置后端 ====================
// 使用静态初始化自动注册

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
