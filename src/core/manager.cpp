#include "core/manager.hpp"

std::unique_ptr<IMemoryResource> MemoryManager::make_resource(Device dev, size_t dev_id) {
    switch (dev) {
        case Device::CPU:
            return std::make_unique<CpuMemoryResource>(lock_memory_);
#ifdef BACKEND_CUDA
        case Device::CUDA:
            return std::make_unique<CudaMemoryResource>(static_cast<int>(dev_id));
#endif
        default:
            throw std::runtime_error(std::format(
                "MemoryManager: unsupported device {}", static_cast<int>(dev)));
    }
}

// 创建存储管理，输入设备，权重使用量，激活使用量上限，kv-cache使用量
DevicePools& MemoryManager::get_or_create(
    Device dev, 
    size_t dev_id,
    size_t weight_cap,
    size_t activation_cap,
    size_t kv_cap)
{
    DevKey key{dev, dev_id};
    auto it = devices_.find(key);
    if (it != devices_.end()) return it->second;
    auto res_w = make_resource(dev, dev_id);
    auto res_a = make_resource(dev, dev_id);
    DevicePools pools;
    if (weight_cap > 0) {
        pools.weight = std::make_unique<MemoryPool>(std::move(res_w), weight_cap, "weight");
    }
    if (activation_cap > 0) {
        pools.activation = std::make_unique<MemoryPool>(std::move(res_a), activation_cap, "activation");
    }
    if (kv_cap > 0) {
        auto res_k = make_resource(dev, dev_id);
        pools.kv_cache = std::make_unique<MemoryPool>(std::move(res_k), kv_cap, "kv_cache");
    }
    auto [inserted, _] = devices_.emplace(key, std::move(pools));
    return inserted->second;
}

DevicePools* MemoryManager::get(Device dev, size_t dev_id) {
    DevKey key{dev, dev_id};
    auto it = devices_.find(key);
    return it != devices_.end() ? &it->second : nullptr;
}

void MemoryManager::reset_all_activations() {
    for (auto& [key, pools] : devices_) {
        pools.reset_activation();
    }
}

void MemoryManager::print_all_usage() const {
    std::println("\n=== Memory Usage ===");
    for (const auto& [key, pools] : devices_) {
        std::println("  {}:{}", device_to_string(key.dev), key.id);
        pools.print_usage();
    }
    std::println("====================\n");
}
