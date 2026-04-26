#pragma once
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include "graph.h"
#include "backend/backend.h"
#include "utils/utils.hpp"
#include "core/manager.h"

class GraphScheduler {
public:
    struct Config {
        size_t vocab_size = 0;           //token表的大小
        size_t kv_cache_per_layer;

        int64_t max_seq_len = 1;         //动态维度(-1)的解析值，用于激活池大小估算

        float top_p = 0.9f;              // 采样时的 top-p 参数
        float temperature = 0.8f;        // 采样时的 temperature 参数
        float memory_headroom;
        float activation_pool_factor;  // 激活内存池大小 = 实际激活内存需求 * activation_pool_factor。比实际需求大一点点。
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
        size_t dev_id = 0;
        size_t total_bytes = 0;
        size_t weight_bytes = 0;
        size_t activation_bytes = 0;
        LayerAssignment() = default;
        LayerAssignment(int s, int e, Device d, size_t id, size_t b, size_t w, size_t a)
            : start_layer(s), end_layer(e), device(d), dev_id(id),
              total_bytes(b), weight_bytes(w), activation_bytes(a) {}
    };
    explicit GraphScheduler(ComputeGraph cg, Config cfg)
        : graph_(std::move(cg)), config_(cfg) {
        mmanager_ = std::make_unique<MemoryManager>();
    }

    void schedule(const std::vector<BackendInfo>& devices);

    const std::vector<LayerAssignment>& get_assignments() const { return assignments_; }
    std::unique_ptr<MemoryManager>& mmanager() { return mmanager_; }
    const std::unique_ptr<MemoryManager>& mmanager() const { return mmanager_; }
    void export_dot(const std::string& path) const { graph_.export_dot(path); }

    const ComputeGraph& graph() const { return graph_; }
    Config& config() { return config_; }
    const Config& config() const { return config_; }

    size_t vocab_size() const { return config_.vocab_size; }
    int64_t max_seq_len() const { return config_.max_seq_len; }
    float top_p() const { return config_.top_p; }
    float temperature() const { return config_.temperature; }
    
private:
    Config config_;
    ComputeGraph graph_;
    std::unique_ptr<MemoryManager> mmanager_;
    std::vector<LayerAssignment> assignments_;

    void assign_global_nodes(ComputeGraph& graph, Device cpu) const;
    void unify_weight_devices(ComputeGraph& graph) const;
    std::vector<LayerCost> estimate_layer_costs(const ComputeGraph& graph) const;
    void apply_assignment(ComputeGraph& graph, const std::vector<LayerAssignment>& assignments) const;
    std::vector<LayerAssignment> assign_layers(const std::vector<LayerCost>& costs,const std::vector<BackendInfo>& devices) const;


    void insert_copy_edges(ComputeGraph& graph) const;
    void create_memory_pools(const ComputeGraph& graph, const std::vector<BackendInfo>& devices);
    void initialize_kv_cache();
    void print_summary(const std::vector<LayerCost>& costs, const std::vector<BackendInfo>& devices) const;

    static std::string format_bytes(size_t bytes) {
        if (bytes >= 1ULL << 30) return std::format("{:.1f} GB", static_cast<double>(bytes) / (1ULL << 30));
        if (bytes >= 1ULL << 20) return std::format("{:.1f} MB", static_cast<double>(bytes) / (1ULL << 20));
        if (bytes >= 1ULL << 10) return std::format("{:.1f} KB", static_cast<double>(bytes) / (1ULL << 10));
        return std::format("{} B", bytes);
    }
};
