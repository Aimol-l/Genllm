#pragma once
#include <cstddef>
#include <memory>
#include <map>

#include "pools.h"
#include "resource.h"
#include "graph.h"
#include "utils/utils.hpp"
#include "core/gguf_parser.h"
#include "core/page_attention.h"

class MemoryManager {
private:
    struct DevKey {
        Device dev = Device::CPU;
        size_t id  = 0;
        bool operator<(const DevKey& o) const {
            return dev < o.dev || (dev == o.dev && id < o.id);
        }
    };
    bool lock_memory_ = false;
    std::map<DevKey, DevicePools> devices_;
    std::map<DevKey, std::unique_ptr<PagedAttentionManager>> attention_managers_;
    
    std::unique_ptr<IMemoryResource> make_resource(Device dev, size_t dev_id);
public:
    MemoryManager();
    explicit MemoryManager(bool lock_memory);
    ~MemoryManager() = default;
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    void reset_all_activations();
    void print_all_usage() const;
    void reset_kv_cache();


    void load_weights(GGUFParser& parser, const ComputeGraph& graph);

    [[nodiscard]] DevicePools* get(Device dev, size_t dev_id);
    DevicePools& get_or_create(Device dev, size_t dev_id,size_t weight_cap,size_t activation_cap,size_t kv_cap = 0);

    [[nodiscard]] PagedAttentionManager* get_attention_manager(Device dev, size_t dev_id);
    PagedAttentionManager& create_attention_manager(Device dev, size_t dev_id);
};

extern MemoryManager* g_mem_manager;