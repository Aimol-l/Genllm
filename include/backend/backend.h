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
// 设备能力描述
struct BackendInfo {
    size_t id;
    Device device;              // CPU/CUDA_0/CUDA_1/SYCL/VULKAN...
    size_t total_memory;        // 总内存/显存（字节）
    size_t reserved_memory;     // 系统/框架预留（字节）
    double compute_power;       // 相对算力（如 TFLOPS），用于负载均衡
    double bandwidth;           // PCIe/NVLink 带宽（GB/s），用于拷贝估算
    size_t available_memory() const {
        return total_memory - reserved_memory;
    }
};

// 算子特征（用于决策）
struct OpFeature {
    Tensor* tensor;             // 代表该算子的输出张量
    OperationType op_type;
    
    // 计算特征
    int64_t flops;              // 估算 FLOPs
    int64_t bytes_read;         // 输入数据量（不含权重）
    int64_t bytes_write;        // 输出数据量
    int64_t weight_bytes;       // 权重大小
    double compute_intensity;   // flops / (bytes_read + bytes_write)
    
    // 依赖特征
    std::vector<Tensor*> input_tensors;
    std::unordered_map<Device, size_t> input_bytes_on_device;  // 各设备上已有输入数据量
    
    // 设备偏好（可选）
    Device preferred_device = Device::AUTO;
};

// 后端提供者：插件式接口
class BackendProvider {
public:
    virtual ~BackendProvider() = default;
    [[nodiscard]] virtual bool is_available() const = 0;
    [[nodiscard]] virtual int get_device_count() const = 0;
    [[nodiscard]] virtual Device get_backend_type() const = 0;
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
    [[nodiscard]] Device get_backend_type() const override { return Device::CPU; }
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
    [[nodiscard]] const BackendInfo* get_device(Device backend, size_t device_id);
};

// ==================== 便利函数 ====================
inline void print_available_devices() {device_manager().print_devices();}
inline DeviceManager& device_manager() {return DeviceManager::instance();}
inline const std::vector<BackendInfo>& get_available_devices() {return device_manager().get_devices();}
inline const BackendInfo* get_device(Device backend, size_t device_id) {return device_manager().get_device(backend, device_id);}

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
