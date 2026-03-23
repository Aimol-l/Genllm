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
    bool enable_auto_partition;        // 是否自动切分图
    bool enable_memory_optimization;   // 是否启用内存优化
    size_t max_temp_memory_per_device; // 每个设备最大临时内存
    bool prefer_gpu;                   // 优先使用GPU（如果有）

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

    // ==================== 辅助方法 ====================

    // 计算节点的拓扑层级
    void compute_tensor_levels(Tensor* root);

    // 判断节点是否可以执行（所有输入都准备好）
    bool is_ready(Tensor* node, const std::unordered_set<Tensor*>& executed);

    // 根据配置选择最佳设备
    const Device* select_device_for_subgraph(const std::vector<Tensor*>& nodes);

public:
    // 构造函数：自动使用 DeviceManager 的设备
    Scheduler(ComputeGraph* graph, const SchedulerConfig& config = SchedulerConfig());
    ~Scheduler();

    // 禁止拷贝
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // ==================== 设备管理 ====================

    // 获取可用设备列表（从 DeviceManager）
    [[nodiscard]] const std::vector<Device>& get_devices() const {
        return *m_devices;
    }

    // 获取指定后端的设备
    [[nodiscard]] std::vector<const Device*> get_devices_by_backend(Backend backend) const;

    // ==================== 图切分 ====================

    // 自动切分计算图到可用设备
    void partition_graph();

    // 手动指定节点到设备（通过设备索引）
    void assign_to_device(Tensor* node, size_t device_index);

    // 手动指定节点到后端（自动选择该后端的第一个设备）
    void assign_to_backend(Tensor* node, Backend backend);

    // 创建子图（根据设备分配）
    void build_subgraphs();

    // ==================== 内存估算 ====================

    // 计算子图的权重内存需求
    size_t calculate_weight_memory(SubGraph& subgraph);

    // 计算子图的临时内存峰值
    size_t calculate_temp_memory(SubGraph& subgraph);

    // 估算整个图的内存需求
    void estimate_memory();

    // ==================== 内存分配 ====================

    // 获取内存管理器
    [[nodiscard]] MemoryManager* get_memory_manager() {
        return m_memory_manager.get();
    }

    // 为子图分配权重内存
    void allocate_weights_for_subgraph(SubGraph& subgraph);

    // 为整个图分配权重内存
    void allocate_all_weights();

    // 为临时张量分配内存
    void allocate_temp_memory(Tensor* tensor);

    // ==================== 执行计划 ====================

    // 生成拓扑排序的执行顺序
    void build_execute_order(Tensor* root);

    // 优化执行计划（考虑设备并行、内存复用等）
    void optimize_execute_order();

    // ==================== 调度执行 ====================

    // 执行单步推理（从输入到输出）
    void execute(Tensor* root);

    // 执行单个节点
    void execute_node(Tensor* node);

    // 清理临时张量内存
    void cleanup_temp_tensors();

    // ==================== 调试信息 ====================

    // 打印子图信息
    void print_subgraphs() const;

    // 打印执行计划
    void print_execute_order() const;

    // 打印内存统计
    void print_memory_stats() const;

    // 打印使用的设备信息
    void print_devices() const;

    // 导出调度计划到 dot 文件
    void export_schedule_dot(const std::string& filename) const;

    // ==================== 访问器 ====================

    [[nodiscard]] const std::vector<SubGraph>& get_subgraphs() const {
        return m_subgraphs;
    }

    [[nodiscard]] const std::vector<ExecuteTask>& get_execute_order() const {
        return m_execute_order;
    }

    [[nodiscard]] size_t get_subgraph_count() const {
        return m_subgraphs.size();
    }

    [[nodiscard]] bool is_partitioned() const {
        return !m_subgraphs.empty();
    }
};

// ==================== 便利函数 ====================

// 创建默认调度器（使用所有可用设备）
inline Scheduler* create_scheduler(ComputeGraph* graph, const SchedulerConfig& config = SchedulerConfig()) {
    return new Scheduler(graph, config);
}

// 创建CPU专用调度器
inline Scheduler* create_cpu_scheduler(ComputeGraph* graph) {
    SchedulerConfig config;
    config.prefer_gpu = false;
    return new Scheduler(graph, config);
}

// 创建GPU专用调度器（如果有GPU）
inline Scheduler* create_gpu_scheduler(ComputeGraph* graph) {
    SchedulerConfig config;
    config.prefer_gpu = true;
    return new Scheduler(graph, config);
}
