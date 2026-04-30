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
    size_t id            = 0;
    enum Device device   = Device::CPU; // CPU/CUDA_0/CUDA_1/SYCL/VULKAN...
    size_t total_memory  = 0;           // 总内存/显存（字节）
    size_t used_memory = 0;             // 已用内存/显存（字节）
    double compute_power   = 0;         // 相对算力（如 TFLOPS），用于负载均衡
    double bandwidth       = 0;         // PCIe/NVLink 带宽（GB/s），用于拷贝估算
    size_t available_memory() const {
        return total_memory - used_memory;
    }
};

// 后端提供者：插件式接口
class BackendProvider {
public:
    virtual ~BackendProvider() = default;
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
    [[nodiscard]] static size_t get_system_memory();
    [[nodiscard]] static size_t get_system_used_memory();
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
                std::print("[{}] {}  Memory: {:.1f} GB", info.id, provider->get_backend_name(),
                    static_cast<double>(info.total_memory) / (1ULL << 30));
            }
            if (provider->get_backend_type() == Device::CPU) {
                auto* cpu = static_cast<const CPUBackendProvider*>(provider.get());
                auto info = provider->get_backend_info(0);
                std::println("");
                std::println("  Device Name:       {}", cpu->get_cpu_name());
                std::println("  Physical Cores:    {}", cpu->get_physical_cores());
                std::println("  Logical Threads:   {}", cpu->get_thread_count());
                std::println("  System Memory:     {:.1f} GB", static_cast<double>(info.total_memory) / (1ULL << 30));
                std::println("  Available Memory:  {:.1f} GB", static_cast<double>(info.available_memory()) / (1ULL << 30));
#ifdef BACKEND_CUDA
            } else if (provider->get_backend_type() == Device::CUDA) {
                std::println("");
                auto* cuda = static_cast<const CUDABackendProvider*>(provider.get());
                for (int i = 0; i < provider->get_device_count(); ++i) {
                    std::println("  Device Name:       {}", CUDABackendProvider::get_device_name(i));
                    std::println("  Compute Capability: {}.{}", CUDABackendProvider::get_sm_version(i) / 10, CUDABackendProvider::get_sm_version(i) % 10);
                    std::println("  Max threads/block:  {}", cuda->get_max_threads_per_block(i));
                    std::println("  Max block dims:     [{}, {}, {}]",
                        cuda->get_max_block_dims(i)[0],
                        cuda->get_max_block_dims(i)[1],
                        cuda->get_max_block_dims(i)[2]);
                    std::println("  Max grid size:      [{}, {}, {}]",
                        cuda->get_max_grid_size(i)[0],
                        cuda->get_max_grid_size(i)[1],
                        cuda->get_max_grid_size(i)[2]);
                    if (i + 1 < provider->get_device_count()) std::println("");
                }
#endif
            } else {
                std::println("");
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
