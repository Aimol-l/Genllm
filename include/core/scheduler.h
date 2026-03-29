#pragma once
#include "graph.hpp"
#include "memory.hpp"
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <print>

class GraphScheduler {
public:
    // ========== 配置结构 ==========
    struct Config {
        // 评分权重
        double weight_compute = 1.0;
        double weight_comm = 5.0;       // 通信开销权重（调高避免 Ping-pong）
        double weight_memory = 3.0;
        double weight_balance = 0.5;    // 负载均衡权重（调低避免过度分散）
        double compute_intensity_threshold = 50.0;  // 计算密度阈值
        size_t small_tensor_threshold = 10 * 1024 * 1024;  // 10MB，小张量避免拷贝惩罚
        std::unordered_set<OperationType> cpu_only_ops = {
            OperationType::OP_TYPE_TOKENIZE,
            OperationType::OP_TYPE_EMBEDDING,
            OperationType::OP_TYPE_ROPE_CACHE,
            OperationType::OP_TYPE_SAMPLING
        };
    };
    MemoryPool* get_pool(Device dev);
    GraphScheduler(Config cfg = {}) : config_(cfg) {}
    void schedule(ComputeGraph& graph, const std::vector<BackendInfo>& devices);
private:
    Config config_;
    std::unordered_map<Device, std::unique_ptr<MemoryPool>> pools_;

    struct DeviceState {
        struct Stats {
            size_t used_memory = 0;
            int64_t assigned_flops = 0;
            int assigned_ops = 0;
        };
        std::unordered_map<Device, BackendInfo> info;
        std::unordered_map<Device, Stats> stats;
        size_t available_memory(Device dev) const {
            auto it = info.find(dev);
            auto sit = stats.find(dev);
            if (it == info.end() || sit == stats.end()) return 0;
            return it->second.available_memory() - sit->second.used_memory;
        }
    };
    struct OpFeature {
        Tensor* tensor = nullptr;
        OperationType op_type = OperationType::OP_TYPE_NONE;
        int64_t flops = 0;
        int64_t bytes_read = 0;
        int64_t bytes_write = 0;
        int64_t weight_bytes = 0;
        double compute_intensity = 0.0;
        std::vector<Tensor*> input_tensors;
        Device preferred_device = Device::AUTO;
    };
    
    int64_t estimate_flops(Tensor* t) const;
    void allocate_memory(ComputeGraph& graph);
    size_t required_memory(const OpFeature& op) const;
    void print_statistics(const DeviceState& dev_state);
    void init_memory_pools(const std::vector<BackendInfo>& devices);
    DeviceState init_device_state(const std::vector<BackendInfo>& devices);
    Tensor* create_memcpy_proxy(Tensor* src, Device dst_dev, DeviceState& dev_state);
    Device find_less_loaded_device(const DeviceState& dev_state, const OpFeature& op);
    std::unordered_map<Tensor*, OpFeature> extract_op_features(const ComputeGraph& graph);
    void assign_to_device(Tensor* t, Device dev, const OpFeature& op, DeviceState& dev_state);
    void migrate_op(Tensor* t, Device old_dev, Device new_dev,DeviceState& dev_state, OpFeature& op);
    void assign_cpu_only_ops(const std::unordered_map<Tensor*, OpFeature>& features,DeviceState& dev_state);
    void greedy_assign(OpFeature& op, DeviceState& dev_state,const std::unordered_map<Tensor*, OpFeature>& all_features);
    void insert_copy_edges(ComputeGraph& graph, std::unordered_map<Tensor*, OpFeature>& features,DeviceState& dev_state);
    bool validate_and_rebalance(ComputeGraph& graph, DeviceState& dev_state,std::unordered_map<Tensor*, OpFeature>& features);
    void rebalance_to_less_loaded(ComputeGraph& graph, DeviceState& dev_state,std::unordered_map<Tensor*, OpFeature>& features);
    void finalize_external_outputs(ComputeGraph& graph, DeviceState& dev_state,std::unordered_map<Tensor*, OpFeature>& features) ;
    double compute_score(const OpFeature& op, Device dev, const DeviceState& dev_state,const std::unordered_map<Tensor*, OpFeature>& all_features);
};