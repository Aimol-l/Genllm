// scheduler.h - 计算图调度器
#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <queue>
#include <print>
#include <format>
#include "graph.h"
#include "tensor.hpp"
#include "backend.h"
#include "memory_manager.h"
#include "utils/utils.hpp"

// 前向声明
class MemoryPool;

// 子图描述
struct SubGraph {
    int id;                            // 子图ID
    std::vector<Tensor*> nodes;        // 计算节点
    std::vector<Tensor*> weights;      // 权重节点（叶子节点）
    const Device* device;              // 分配的设备（指向 DeviceManager 的设备）
    size_t weight_memory;              // 权重内存需求（字节）
    size_t temp_memory;                // 临时张量内存峰值（字节）
    std::vector<Tensor*> inputs;       // 子图输入（来自其他子图）
    std::vector<Tensor*> outputs;      // 子图输出（到其他子图）
    SubGraph() : id(-1), device(nullptr), weight_memory(0), temp_memory(0) {}
    // 获取设备字符串
    [[nodiscard]] std::string device_str() const {
        return device ? device->to_string() : "None";
    }
};

// 执行任务
struct ExecuteTask {
    Tensor* node;                      // 要执行的计算节点
    SubGraph* subgraph;                // 所属子图
    int priority;                      // 优先级（拓扑序）
    ExecuteTask(Tensor* n, SubGraph* sg, int p)
        : node(n), subgraph(sg), priority(p) {}
    // 优先级队列比较（优先级高的先执行）
    bool operator<(const ExecuteTask& other) const {
        return priority < other.priority;
    }
};

// 调度器配置
struct SchedulerConfig {
    bool prefer_gpu;                   // 优先使用GPU（如果有）
    bool enable_auto_partition;        // 是否自动切分图
    bool enable_memory_optimization;   // 是否启用内存优化
    size_t max_temp_memory_per_device; // 每个设备最大临时内存
    SchedulerConfig()
        : enable_auto_partition(true)
        , enable_memory_optimization(true)
        , max_temp_memory_per_device(1ULL << 30)  // 默认1GB
        , prefer_gpu(true) {}
};

// 调度器类
class Scheduler {
private:
    ComputeGraph* m_graph;
    const std::vector<Device>* m_devices;  // 引用 DeviceManager 的设备列表
    std::vector<SubGraph> m_subgraphs;
    SchedulerConfig m_config;

    // 执行相关
    std::vector<ExecuteTask> m_execute_order;  // 拓扑排序的执行顺序
    std::unordered_map<Tensor*, int> m_tensor_levels;  // 节点层级

    // 内存管理
    std::unique_ptr<MemoryManager> m_memory_manager;
    void compute_tensor_levels(Tensor* root);
    bool is_ready(Tensor* node, const std::unordered_set<Tensor*>& executed);
    const Device* select_device_for_subgraph(const std::vector<Tensor*>& nodes);

public:
    ~Scheduler();
    Scheduler(ComputeGraph* graph, const SchedulerConfig& config = SchedulerConfig());

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    [[nodiscard]] const std::vector<Device>& get_devices() const {
        return *m_devices;
    }
    [[nodiscard]] std::vector<const Device*> get_devices_by_backend(Backend backend) const;
    void partition_graph();
    void assign_to_device(Tensor* node, size_t device_index);
    void assign_to_backend(Tensor* node, Backend backend);
    void build_subgraphs();
    size_t calculate_weight_memory(SubGraph& subgraph);
    size_t calculate_temp_memory(SubGraph& subgraph);
    void estimate_memory();

    [[nodiscard]] MemoryManager* get_memory_manager() {return m_memory_manager.get();}
    void allocate_weights_for_subgraph(SubGraph& subgraph);
    void allocate_all_weights();
    void allocate_temp_memory(Tensor* tensor);
    void build_execute_order(Tensor* root);
    void optimize_execute_order();
    void execute(Tensor* root);
    void execute_node(Tensor* node);
    void cleanup_temp_tensors();
    void print_subgraphs() const;
    void print_execute_order() const;
    void print_memory_stats() const;
    void print_devices() const;
    void export_schedule_dot(const std::string& filename) const;

    [[nodiscard]] const std::vector<SubGraph>& get_subgraphs() const {return m_subgraphs;}
    [[nodiscard]] const std::vector<ExecuteTask>& get_execute_order() const {return m_execute_order;}
    [[nodiscard]] size_t get_subgraph_count() const {return m_subgraphs.size();}
    [[nodiscard]] bool is_partitioned() const {return !m_subgraphs.empty();}
};