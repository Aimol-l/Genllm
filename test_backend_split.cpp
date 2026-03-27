// 测试后端拆分是否成功
#include "backend/backend.h"
#include <iostream>

int main() {
    std::cout << "=== Testing Backend Split ===" << std::endl;

    // 初始化后端
    backend::init_builtin_backends();

    // 测试单例模式
    auto& dev_mgr = device_manager();

    // 测试获取所有设备
    auto devices = get_available_devices();
    std::cout << "Found " << devices.size() << " devices" << std::endl;

    // 测试打印设备信息
    print_available_devices();

    // 测试获取 CPU 设备
    auto cpu = get_cpu_device();
    if (cpu) {
        std::cout << "CPU device: " << cpu->to_string() << std::endl;
    }

    // 测试检查是否有 CUDA
    std::cout << "Has CUDA: " << (has_cuda() ? "Yes" : "No") << std::endl;

    // 测试获取设备摘要
    std::cout << "Summary: " << get_device_summary() << std::endl;

    return 0;
}
