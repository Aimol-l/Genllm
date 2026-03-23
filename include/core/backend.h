// backend.h - 后端设备管理
#pragma once
#include <vector>
#include <memory>
#include <print>
#include "utils/utils.hpp"

// 编译时后端支持检测
#if defined(USE_CUDA)
    #define GENLLM_HAS_CUDA 1
#else
    #define GENLLM_HAS_CUDA 0
#endif

#if defined(USE_VULKAN)
    #define GENLLM_HAS_VULKAN 1
#else
    #define GENLLM_HAS_VULKAN 0
#endif

#if defined(USE_SYCL)
    #define GENLLM_HAS_SYCL 1
#else
    #define GENLLM_HAS_SYCL 0
#endif

// 总是有 CPU 后端
#define GENLLM_HAS_CPU 1

// 设备描述
struct Device {
    Backend backend;
    size_t device_id;
    size_t total_memory;        // 总内存（字节）
    std::string name;           // 设备名称

    Device(Backend be, size_t id, size_t memory, const std::string& n = "")
        : backend(be), device_id(id), total_memory(memory), name(n) {}

    [[nodiscard]] std::string to_string() const {
        return std::format("{}:{} ({} MB)",
            backend_to_string(backend),
            device_id,
            total_memory / (1024 * 1024));
    }

private:
    [[nodiscard]] static const char* backend_to_string(Backend be) {
        switch (be) {
            case Backend::CPU:    return "CPU";
            case Backend::CUDA:   return "CUDA";
            case Backend::Vulkan: return "Vulkan";
            case Backend::Sycl:   return "Sycl";
            default:              return "Unknown";
        }
    }
};

// 设备管理器（自动检测可用设备）
class DeviceManager {
private:
    std::vector<Device> m_devices;
    static std::unique_ptr<DeviceManager> s_instance;

    DeviceManager() {
        detect_devices();
    }

    // 检测可用设备
    void detect_devices() {
        // CPU 后端总是可用
        size_t cpu_memory = get_system_memory();
        m_devices.emplace_back(Backend::CPU, 0, cpu_memory, "CPU");

#if GENLLM_HAS_CUDA
        // 检测 CUDA 设备
        int cuda_device_count = get_cuda_device_count();
        for (int i = 0; i < cuda_device_count; ++i) {
            size_t memory = get_cuda_device_memory(i);
            m_devices.emplace_back(Backend::CUDA, i, memory,
                std::format("CUDA-{}", i));
        }
        std::println("Detected {} CUDA device(s)", cuda_device_count);
#endif

#if GENLLM_HAS_VULKAN
        // 检测 Vulkan 设备
        int vulkan_device_count = get_vulkan_device_count();
        for (int i = 0; i < vulkan_device_count; ++i) {
            size_t memory = get_vulkan_device_memory(i);
            m_devices.emplace_back(Backend::Vulkan, i, memory,
                std::format("Vulkan-{}", i));
        }
        std::println("Detected {} Vulkan device(s)", vulkan_device_count);
#endif

#if GENLLM_HAS_SYCL
        // 检测 Sycl 设备
        int sycl_device_count = get_sycl_device_count();
        for (int i = 0; i < sycl_device_count; ++i) {
            size_t memory = get_sycl_device_memory(i);
            m_devices.emplace_back(Backend::Sycl, i, memory,
                std::format("Sycl-{}", i));
        }
        std::println("Detected {} Sycl device(s)", sycl_device_count);
#endif

        std::println("Total devices: {}", m_devices.size());
    }

    // ==================== 平台特定的设备检测 ====================

    // 获取系统内存（CPU）
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

#if GENLLM_HAS_CUDA
    [[nodiscard]] static int get_cuda_device_count() {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }

    [[nodiscard]] static size_t get_cuda_device_memory(int device_id) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        return prop.totalGlobalMem;
    }
#endif

#if GENLLM_HAS_VULKAN
    [[nodiscard]] static int get_vulkan_device_count() {
        // TODO: 实现 Vulkan 设备检测
        return 0;
    }

    [[nodiscard]] static size_t get_vulkan_device_memory(int device_id) {
        // TODO: 实现 Vulkan 内存查询
        return 0;
    }
#endif

#if GENLLM_HAS_SYCL
    [[nodiscard]] static int get_sycl_device_count() {
        // TODO: 实现 Sycl 设备检测
        return 0;
    }

    [[nodiscard]] static size_t get_sycl_device_memory(int device_id) {
        // TODO: 实现 Sycl 内存查询
        return 0;
    }
#endif

public:
    // 获取单例实例
    static DeviceManager& get_instance() {
        if (!s_instance) {
            s_instance = std::make_unique<DeviceManager>();
        }
        return *s_instance;
    }

    // 禁止拷贝和移动
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    DeviceManager(DeviceManager&&) = delete;
    DeviceManager& operator=(DeviceManager&&) = delete;

    ~DeviceManager() = default;

    // ==================== 设备查询 ====================

    // 获取所有设备
    [[nodiscard]] const std::vector<Device>& get_devices() const {
        return m_devices;
    }

    // 获取设备数量
    [[nodiscard]] size_t device_count() const {
        return m_devices.size();
    }

    // 获取指定后端的设备
    [[nodiscard]] std::vector<const Device*> get_devices_by_backend(Backend backend) const {
        std::vector<const Device*> result;
        for (const auto& dev : m_devices) {
            if (dev.backend == backend) {
                result.push_back(&dev);
            }
        }
        return result;
    }

    // 获取 CPU 设备
    [[nodiscard]] const Device* get_cpu_device() const {
        for (const auto& dev : m_devices) {
            if (dev.backend == Backend::CPU) {
                return &dev;
            }
        }
        return nullptr;
    }

    // 获取指定后端的第一个设备
    [[nodiscard]] const Device* get_first_device(Backend backend) const {
        auto devs = get_devices_by_backend(backend);
        return devs.empty() ? nullptr : devs[0];
    }

    // 检查是否有某个后端
    [[nodiscard]] bool has_backend(Backend backend) const {
        return !get_devices_by_backend(backend).empty();
    }

    // ==================== 调试信息 ====================

    // 打印所有设备信息
    void print_devices() const {
        std::println("\n=== Available Devices ({}) ===", m_devices.size());
        for (const auto& dev : m_devices) {
            std::println("  {}", dev.to_string());
        }
        // 打印编译时配置
        std::println("\nCompile-time backend support:");
        std::println("  CPU:    {}", GENLLM_HAS_CPU ? "Yes" : "No");
        std::println("  CUDA:   {}", GENLLM_HAS_CUDA ? "Yes" : "No");
        std::println("  Vulkan: {}", GENLLM_HAS_VULKAN ? "Yes" : "No");
        std::println("  Sycl:   {}", GENLLM_HAS_SYCL ? "Yes" : "No");
    }
};

// 静态成员初始化
inline std::unique_ptr<DeviceManager> DeviceManager::s_instance = nullptr;

// 便利函数：获取设备管理器
inline DeviceManager& device_manager() {
    return DeviceManager::get_instance();
}

// 便利函数：获取所有可用设备
inline const std::vector<Device>& get_available_devices() {
    return device_manager().get_devices();
}

// 便利函数：打印设备信息
inline void print_available_devices() {
    device_manager().print_devices();
}
