#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <print>
#include <format>
#include <fstream>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif
#include "utils/utils.hpp"

// ==================== 后端提供者接口 ====================

// 后端设备信息
struct BackendInfo {
    Backend backend;
    size_t device_id;
    size_t total_memory;
    std::string name;

    // 构造函数
    BackendInfo() : backend(Backend::CPU), device_id(0), total_memory(0), name("Unknown") {}
    BackendInfo(Backend b, size_t id, size_t mem, const std::string& n): backend(b), device_id(id), total_memory(mem), name(n) {}
    size_t get_free_memory() const;
};

// 后端提供者：插件式接口
class BackendProvider {
public:
    virtual ~BackendProvider() = default;
    [[nodiscard]] virtual bool is_available() const = 0;
    [[nodiscard]] virtual int get_device_count() const = 0;
    [[nodiscard]] virtual Backend get_backend_type() const = 0;
    [[nodiscard]] virtual const char* get_backend_name() const = 0;
    [[nodiscard]] virtual BackendInfo get_device_info(int device_id) const = 0;
};

// ==================== 后端注册系统 ====================
class BackendRegistry {
private:
    BackendRegistry() = default;
    std::vector<std::unique_ptr<BackendProvider>> providers_;
public:
    static BackendRegistry& instance();
    void register_provider(std::unique_ptr<BackendProvider> provider);
    [[nodiscard]] const std::vector<std::unique_ptr<BackendProvider>>& get_providers() const {return providers_;}
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
    [[nodiscard]] BackendInfo get_device_info(int device_id) const override;
private:
    [[nodiscard]] static size_t get_system_memory();
    [[nodiscard]] static std::string get_cpu_name();
};

// CUDA 后端
#if defined(USE_CUDA)
#include <cuda_runtime.h>

class CUDABackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] int get_device_count() const override;
    [[nodiscard]] BackendInfo get_device_info(int device_id) const override;
    [[nodiscard]] Backend get_backend_type() const override { return Backend::CUDA; }
    [[nodiscard]] const char* get_backend_name() const override { return "CUDA"; }
};
#endif

// Vulkan 后端
#if defined(USE_VULKAN)
#include <vulkan/vulkan.h>

class VulkanBackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] int get_device_count() const override;
    [[nodiscard]] BackendInfo get_device_info(int device_id) const override;
    [[nodiscard]] Backend get_backend_type() const override { return Backend::Vulkan; }
    [[nodiscard]] const char* get_backend_name() const override { return "Vulkan"; }
};
#endif

// Sycl 后端
#if defined(USE_SYCL)
#include <sycl/sycl.hpp>

class SyclBackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] int get_device_count() const override;
    [[nodiscard]] BackendInfo get_device_info(int device_id) const override;
    [[nodiscard]] Backend get_backend_type() const override { return Backend::Sycl; }
    [[nodiscard]] const char* get_backend_name() const override { return "Sycl"; }
};
#endif

// ==================== 设备管理器 ====================

class DeviceManager {
private:
    bool initialized_ = false;
    std::vector<BackendInfo> devices_;

    DeviceManager() = default;
    void detect_devices();
public:
    static DeviceManager& instance();
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    void print_devices();
    [[nodiscard]] size_t device_count();
    [[nodiscard]] const std::vector<BackendInfo>& get_devices();
    [[nodiscard]] const BackendInfo* get_device(Backend backend, size_t device_id);
};

// ==================== 便利函数 ====================
inline void print_available_devices() {device_manager().print_devices();}
inline DeviceManager& device_manager() {return DeviceManager::instance();}
inline const std::vector<BackendInfo>& get_available_devices() {return device_manager().get_devices();}
inline const BackendInfo* get_device(Backend backend, size_t device_id) {return device_manager().get_device(backend, device_id);}

#define REGISTER_BACKEND(ProviderClass) \
    static struct ProviderClass##Registrar { \
        ProviderClass##Registrar() { \
            BackendRegistry::instance().register_provider( \
                std::make_unique<ProviderClass>()); \
        } \
    } g_##ProviderClass##_registrar;

// ==================== 后端初始化函数 ====================
namespace backend {
    // 初始化所有内置后端
    void init_builtin_backends();
}
