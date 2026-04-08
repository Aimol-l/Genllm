#include "backend/backend.h"
#include <fstream>
#include <unordered_set>

#if defined(_WIN32)
#include <windows.h>
#endif

BackendInfo CPUBackendProvider::get_backend_info(int device_id) const {
    size_t memory = get_system_memory();
    return BackendInfo{0, Device::CPU, memory, 2ULL << 30, 20, 0};
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
        return 8ULL << 30;
    #else
        return 8ULL << 30;
    #endif
}

std::string CPUBackendProvider::get_cpu_name() {
#if defined(_WIN32)
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

unsigned int CPUBackendProvider::get_thread_count() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
#elif defined(__linux__) || defined(__APPLE__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<unsigned int>(n) : 1;
#else
    return 1;
#endif
}

unsigned int CPUBackendProvider::get_physical_cores() {
#if defined(__linux__)
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::unordered_set<std::string> cores;
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("core id") == 0) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                cores.insert(line.substr(pos + 2));
            }
        }
    }
    return cores.empty() ? get_thread_count() : static_cast<unsigned int>(cores.size());
#elif defined(__APPLE__)
    return get_thread_count();
#elif defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    DWORD length = 0;
    GetLogicalProcessorInformation(NULL, &length);
    if (length == 0) return si.dwNumberOfProcessors;
    auto buffer = std::make_unique<char[]>(length);
    auto info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(buffer.get());
    if (!GetLogicalProcessorInformation(info, &length)) return si.dwNumberOfProcessors;
    unsigned int count = 0;
    size_t offset = 0;
    while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION) <= length) {
        if (info->Relationship == RelationProcessorCore) count++;
        offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        info++;
    }
    return count > 0 ? count : si.dwNumberOfProcessors;
#else
    return 1;
#endif
}


static struct CPUBackendProviderRegistrar {
    CPUBackendProviderRegistrar() {
        BackendRegistry::instance().register_provider(std::make_unique<CPUBackendProvider>());
    }
} g_cpu_backend_registrar;