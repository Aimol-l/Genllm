#include "backend/backend.h"

size_t BackendInfo::get_free_memory() const {
    // TODO: 实现获取可用内存的逻辑
    return total_memory;  // 暂时返回总内存
}

// ==================== CPU 后端实现 ====================

BackendInfo CPUBackendProvider::get_device_info(int device_id) const {
    size_t memory = get_system_memory();
    return BackendInfo(Backend::CPU, 0, memory, get_cpu_name());
}

size_t CPUBackendProvider::get_system_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return static_cast<size_t>(status.ullTotalPhys);
#elif defined(__linux__) || defined(__APPLE__)
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
    return 8ULL << 30;  // 默认 8GB
#else
    return 8ULL << 30;  // 默认 8GB
#endif
}

std::string CPUBackendProvider::get_cpu_name() {
#if defined(_WIN32)
    // Windows: 使用注册表获取 CPU 名称
    HKEY hKey;
    char cpuName[256] = "Unknown CPU";
    DWORD size = sizeof(cpuName);
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
            (LPBYTE)cpuName, &size);
        RegCloseKey(hKey);
    }
    return cpuName;
#elif defined(__linux__)
    // Linux: 从 /proc/cpuinfo 读取
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") == 0) {
            size_t pos = line.find(':');
            if (pos != std::string::npos && pos + 2 < line.length()) {
                return line.substr(pos + 2);
            }
        }
    }
    return "Unknown CPU";
#elif defined(__APPLE__)
    // macOS: 使用 sysctl
    char cpuName[256];
    size_t len = sizeof(cpuName);
    if (sysctlbyname("machdep.cpu.brand_string", cpuName, &len, NULL, 0) == 0) {
        return cpuName;
    }
    return "Unknown CPU";
#else
    return "Unknown CPU";
#endif
}
