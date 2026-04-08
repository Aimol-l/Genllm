#pragma once
#include <cstddef>
#include <memory>
#include <unordered_map>

#include "pools.h"
#include "resource.h"
#include "utils/utils.hpp"

class MemoryManager {
private:
    struct DevKey {
        Device dev;
        size_t id;
        bool operator==(const DevKey& o) const { return dev == o.dev && id == o.id; }
    };
    struct DevKeyHash {
        size_t operator()(const DevKey& k) const {
            return static_cast<size_t>(k.dev) ^ (k.id << 8);
        }
    };
    std::unordered_map<DevKey, DevicePools, DevKeyHash> devices_;
    
    bool lock_memory_ = false;
    std::unique_ptr<IMemoryResource> make_resource(Device dev, size_t dev_id);
public:
    MemoryManager() = default;
    MemoryManager(bool lock_memory) : lock_memory_(lock_memory) {}
    ~MemoryManager() = default;
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    void reset_all_activations();
    void print_all_usage() const;
    DevicePools* get(Device dev, size_t dev_id);
    DevicePools& get_or_create(Device dev, size_t dev_id,
                               size_t weight_cap,
                               size_t activation_cap,
                               size_t kv_cap = 0);
};
