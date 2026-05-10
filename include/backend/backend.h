#pragma once
#include <vector>
#include <memory>
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif
#if defined(_WIN32)
#include <windows.h>
#endif

#include <print>
#include "utils/utils.hpp"

// ==================== 后端提供者接口 ====================
// 设备能力描述
struct BackendInfo {
    size_t id            = 0;           // 设备ID
    size_t total_memory  = 0;           // 总内存/显存（字节）
    size_t used_memory   = 0;           // 已用内存/显存（字节）
    double compute_power = 0;           // 相对算力（如 TFLOPS），用于负载均衡
    double bandwidth     = 0;           // PCIe/NVLink 带宽（GB/s），用于拷贝估算
    enum Device device   = Device::CPU; // CPU/CUDA/SYCL/VULKAN...
    size_t available_memory() const {
        return total_memory - used_memory;
    }
};

// 后端提供者：插件式接口
class BackendProvider {
public:
    virtual ~BackendProvider() = default;
    virtual void print_device_info(int device_id) const = 0;
    [[nodiscard]] virtual bool is_available() const = 0;
    [[nodiscard]] virtual int get_device_count() const = 0;
    [[nodiscard]] virtual Device get_backend_type() const = 0;
    [[nodiscard]] virtual const char* get_backend_name() const = 0;
    [[nodiscard]] virtual BackendInfo get_backend_info(int device_id) const = 0;
};

// ==================== 后端注册系统 ====================
class BackendRegistry {
private:
    BackendRegistry() = default;
    std::vector<std::unique_ptr<BackendProvider>> providers_;
public:
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    static BackendRegistry& instance(){
        static BackendRegistry registry;
        return registry;
    }
    void register_provider(std::unique_ptr<BackendProvider> provider){
        if (provider && provider->is_available()) {
            providers_.push_back(std::move(provider));
        }
    }
    [[nodiscard]] const std::vector<std::unique_ptr<BackendProvider>>& get_providers() const {
        return providers_;
    }
};

// ==================== 内置后端实现 ====================
// CPU 后端（总是可用）
class CPUBackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override { return true; }
    [[nodiscard]] int get_device_count() const override { return 1; }
    [[nodiscard]] Device get_backend_type() const override { return Device::CPU; }
    [[nodiscard]] const char* get_backend_name() const override { return "CPU"; }
    [[nodiscard]] BackendInfo get_backend_info(int device_id) const override;
    void print_device_info(int device_id) const override;
    [[nodiscard]] static size_t get_system_memory();
    [[nodiscard]] static size_t get_system_available_memory();
    [[nodiscard]] static std::string get_cpu_name();
    [[nodiscard]] static unsigned int get_physical_cores();
    [[nodiscard]] static unsigned int get_thread_count();
};

#ifdef BACKEND_CUDA
class CUDABackendProvider : public BackendProvider {
public:
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] int get_device_count() const override;
    [[nodiscard]] Device get_backend_type() const override { return Device::CUDA; }
    [[nodiscard]] const char* get_backend_name() const override { return "CUDA"; }
    [[nodiscard]] BackendInfo get_backend_info(int device_id) const override;
    [[nodiscard]] static std::string get_device_name(int device_id);
    [[nodiscard]] static int get_sm_version(int device_id);
    [[nodiscard]] static int get_max_threads_per_block(int device_id);
    [[nodiscard]] static std::array<int, 3> get_max_block_dims(int device_id);
    [[nodiscard]] static std::array<int, 3> get_max_grid_size(int device_id);
    void print_device_info(int device_id) const;

};
#endif

#ifdef BACKEND_VULKAN
class VulkanContext;
class VulkanBackendProvider : public BackendProvider {
public:
    VulkanBackendProvider();
    [[nodiscard]] bool is_available() const override;
    [[nodiscard]] int get_device_count() const override;
    [[nodiscard]] Device get_backend_type() const override { return Device::VULKAN; }
    [[nodiscard]] const char* get_backend_name() const override { return "Vulkan"; }
    [[nodiscard]] BackendInfo get_backend_info(int device_id) const override;
    void print_device_info(int device_id) const override;
private:
    VulkanContext* ctx_ = nullptr;
    bool available_ = false;
};
#endif

// ==================== 设备管理器 ====================
class DeviceManager {
private:
    mutable bool initialized_ = false;
    mutable std::vector<BackendInfo> devices_;

    DeviceManager() = default;
    void detect_devices() const {
        devices_.clear();
        auto& registry = BackendRegistry::instance();
        for (const auto& provider : registry.get_providers()) {
            int count = provider->get_device_count();
            for (int i = 0; i < count; ++i) {
                auto info = provider->get_backend_info(i);
                devices_.emplace_back(info);
            }
        }
        initialized_ = true;
    }
public:
    static DeviceManager& instance(){
        static DeviceManager manager;
        return manager;
    }
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    void print_devices() const {
        auto& registry = BackendRegistry::instance();
        std::println("======================= Detected Devices =======================");
        for (const auto& provider : registry.get_providers()) {
            for (int i = 0; i < provider->get_device_count(); ++i) {
                auto info = provider->get_backend_info(i);
                std::println("[{}] {}  Memory: {:.1f} GB", info.id, provider->get_backend_name(),static_cast<double>(info.total_memory) / (1ULL << 30));
                provider->print_device_info(i);
            }
        }
        std::println("================================================================");
    }
    [[nodiscard]] size_t device_count() const { return get_devices().size(); }
    [[nodiscard]] const std::vector<BackendInfo>& get_devices() const {
        if (!initialized_) detect_devices();
        return devices_;
    }
    [[nodiscard]] const BackendInfo* get_device(Device dev, size_t device_id) const {
        for (const auto& back : get_devices()) {
            if (back.device == dev && back.id == device_id) {
                return &back;
            }
        }
        return nullptr;
    }
};
