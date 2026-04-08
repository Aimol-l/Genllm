#pragma once
#include <memory>
#include <vector>
#include "graph.hpp"
#include "backend/backend.h"
#include "utils/utils.hpp"
#include "core/manager.hpp"

class GraphScheduler {
public:
    struct Config {
        double memory_headroom;
        size_t kv_cache_per_layer;
        size_t activation_pool_factor;
        Config(double head = 0.1, size_t kv = 0, size_t act_factor = 2)
            : memory_headroom(head), kv_cache_per_layer(kv), activation_pool_factor(act_factor) {}
    };
    struct LayerCost {
        int layer_id = -1;
        size_t weight_bytes = 0;
        size_t activation_bytes = 0;
        size_t kv_cache_bytes = 0;
        size_t total() const { return weight_bytes + activation_bytes + kv_cache_bytes; }
    };
    struct LayerAssignment {
        int start_layer = 0;
        int end_layer = 0;
        Device device = Device::CPU;
        size_t total_bytes = 0;
        size_t weight_bytes = 0;
        size_t activation_bytes = 0;
        LayerAssignment() = default;
        LayerAssignment(int s, int e, Device d, size_t b, size_t w, size_t a)
            : start_layer(s), end_layer(e), device(d),
              total_bytes(b), weight_bytes(w), activation_bytes(a) {}
    };
    explicit GraphScheduler(ComputeGraph cg, Config cfg = {})
        : graph_(std::move(cg)), config_(cfg) {
        memory_ = std::make_unique<MemoryManager>();
    }

    void schedule(const std::vector<BackendInfo>& devices);
    const std::vector<LayerAssignment>& get_assignments() const { return assignments_; }
    std::unique_ptr<MemoryManager>& memory() { return memory_; }
    const std::unique_ptr<MemoryManager>& memory() const { return memory_; }
    void export_dot(const std::string& path) const { graph_.export_dot(path); }

private:
    Config config_;
    ComputeGraph graph_;
    std::unique_ptr<MemoryManager> memory_;
    std::vector<LayerAssignment> assignments_;

    void assign_global_nodes(ComputeGraph& graph, Device cpu) const;
    void unify_weight_devices(ComputeGraph& graph) const;
    std::vector<LayerCost> estimate_layer_costs(const ComputeGraph& graph) const;
    void apply_assignment(ComputeGraph& graph, const std::vector<LayerAssignment>& assignments) const;
    std::vector<LayerAssignment> assign_layers(const std::vector<LayerCost>& costs,const std::vector<BackendInfo>& devices) const;


    void insert_copy_edges(ComputeGraph& graph) const;
    void create_memory_pools(const ComputeGraph& graph, const std::vector<BackendInfo>& devices);
    void print_summary(const std::vector<LayerCost>& costs, const std::vector<BackendInfo>& devices) const;

    static std::string format_bytes(size_t bytes) {
        if (bytes >= 1ULL << 30) return std::format("{:.1f} GB", static_cast<double>(bytes) / (1ULL << 30));
        if (bytes >= 1ULL << 20) return std::format("{:.1f} MB", static_cast<double>(bytes) / (1ULL << 20));
        if (bytes >= 1ULL << 10) return std::format("{:.1f} KB", static_cast<double>(bytes) / (1ULL << 10));
        return std::format("{} B", bytes);
    }
};
