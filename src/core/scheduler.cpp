// scheduler.cpp - 调度器实现
#include "core/scheduler.h"
#include <algorithm>
#include <queue>
#include <fstream>

// 构造函数
Scheduler::Scheduler(ComputeGraph* graph, const SchedulerConfig& config)
    : m_graph(graph)
    , m_devices(&device_manager().get_devices())  // 使用 DeviceManager 的设备
    , m_config(config)
    , m_memory_manager(std::make_unique<MemoryManager>()) {

    // 打印可用设备
    print_devices();
}

// 析构函数
Scheduler::~Scheduler() = default;

// ==================== 设备管理 ====================

std::vector<const Device*> Scheduler::get_devices_by_backend(Backend backend) const {
    std::vector<const Device*> result;
    for (const auto& dev : *m_devices) {
        if (dev.backend == backend) {
            result.push_back(&dev);
        }
    }
    return result;
}

void Scheduler::print_devices() const {
    std::println("\n=== Scheduler Using Devices ===");
    for (const auto& dev : *m_devices) {
        std::println("  {}", dev.to_string());
    }
}

// ==================== 图切分 ====================

void Scheduler::partition_graph() {
    if (!m_graph || m_graph->m_nodes.empty()) {
        std::println("Warning: Empty graph, nothing to partition");
        return;
    }

    std::println("Partitioning computation graph...");

    // 简单策略：按层级切分到不同设备
    // TODO: 实现更智能的切分算法（考虑计算量、内存需求、通信成本等）

    if (m_devices.empty()) {
        std::println("Warning: No devices available, using CPU");
        add_device(Backend::CPU, 0, 8ULL << 30);  // 默认8GB
    }

    // 计算所有节点的层级
    if (!m_graph->m_nodes.empty()) {
        // 假设最后一个节点是输出节点
        compute_tensor_levels(m_graph->m_nodes.back());
    }

    // 按层级分组
    std::unordered_map<int, std::vector<Tensor*>> level_groups;
    std::unordered_map<Tensor*, int> node_to_level;

    for (auto* node : m_graph->m_nodes) {
        int level = m_tensor_levels[node];
        level_groups[level].push_back(node);
        node_to_level[node] = level;
    }

    // 将层级分配到设备（轮询）
    int num_devices = static_cast<int>(m_devices.size());
    int subgraph_id = 0;

    for (auto& [level, nodes] : level_groups) {
        SubGraph subgraph;
        subgraph.id = subgraph_id++;
        subgraph.nodes = nodes;
        subgraph.device = &m_devices[subgraph_id % num_devices];

        // 找出这些节点需要的权重
        std::unordered_set<Tensor*> weight_set;
        for (auto* node : nodes) {
            // 检查所有输入，找出叶子节点（权重）
            for (auto* src : node->src) {
                if (src && std::find(m_graph->m_leafs.begin(), m_graph->m_leafs.end(), src) != m_graph->m_leafs.end()) {
                    weight_set.insert(src);
                }
            }
        }
        subgraph.weights.assign(weight_set.begin(), weight_set.end());

        m_subgraphs.push_back(std::move(subgraph));
    }

    std::println("  Partitioned into {} subgraphs", m_subgraphs.size());

    // 估算内存
    estimate_memory();
}

void Scheduler::assign_to_device(Tensor* node, Backend backend, size_t device_id) {
    // 找到对应的设备
    DeviceInfo* target_device = nullptr;
    for (auto& dev : m_devices) {
        if (dev.backend == backend && dev.device_id == device_id) {
            target_device = &dev;
            break;
        }
    }

    if (!target_device) {
        throw std::runtime_error(std::format("Device not found: {}:{}",
            static_cast<int>(backend), device_id));
    }

    // 标记节点（通过在节点名称中添加设备信息）
    if (!node->name.empty()) {
        node->name += std::format("@{}:{}", static_cast<int>(backend), device_id);
    }
}

void Scheduler::build_subgraphs() {
    // 基于设备分配构建子图
    std::unordered_map<std::string, SubGraph> device_subgraphs;

    for (auto* node : m_graph->m_nodes) {
        // 从节点名称提取设备信息（如果有）
        std::string device_key;
        if (node->name.find('@') != std::string::npos) {
            size_t at_pos = node->name.find('@');
            device_key = node->name.substr(at_pos);
        } else {
            device_key = "@CPU:0";  // 默认设备
        }

        if (device_subgraphs.find(device_key) == device_subgraphs.end()) {
            SubGraph sg;
            sg.id = static_cast<int>(device_subgraphs.size());
            device_subgraphs[device_key] = sg;
        }

        device_subgraphs[device_key].nodes.push_back(node);
    }

    m_subgraphs.clear();
    for (auto& [key, sg] : device_subgraphs) {
        m_subgraphs.push_back(std::move(sg));
    }

    estimate_memory();
}

// ==================== 内存估算 ====================

size_t Scheduler::calculate_weight_memory(SubGraph& subgraph) {
    size_t total = 0;
    for (auto* weight : subgraph.weights) {
        if (!weight) continue;

        // 计算张量大小
        size_t num_elements = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && weight->dims[i] != 0; ++i) {
            num_elements *= weight->dims[i];
        }
        total += num_elements * data_type_size(weight->dtype);
    }
    return total;
}

size_t Scheduler::calculate_temp_memory(SubGraph& subgraph) {
    size_t peak_memory = 0;
    size_t current_memory = 0;

    // 模拟执行，追踪内存使用
    // TODO: 更精确的分析（考虑生命周期）

    for (auto* node : subgraph.nodes) {
        if (!node) continue;

        // 节点输出需要的内存
        size_t node_size = 0;
        for (int i = 0; i < TENSOR_MAX_DIMS && node->dims[i] != 0; ++i) {
            if (node_size == 0) {
                node_size = 1;
            }
            node_size *= node->dims[i];
        }
        node_size *= data_type_size(node->dtype);

        current_memory += node_size;
        peak_memory = std::max(peak_memory, current_memory);

        // 如果节点的输入不再被需要，可以释放
        // 这里简化处理：假设输入可以立即释放
        // 实际需要分析数据依赖
    }

    return peak_memory;
}

void Scheduler::estimate_memory() {
    for (auto& subgraph : m_subgraphs) {
        subgraph.weight_memory = calculate_weight_memory(subgraph);
        subgraph.temp_memory = calculate_temp_memory(subgraph);

        std::println("  SubGraph {}: device={}, weight_memory={} MB, temp_memory={} MB",
            subgraph.id,
            subgraph.device ? static_cast<int>(subgraph.device->backend) : -1,
            subgraph.weight_memory / (1024 * 1024),
            subgraph.temp_memory / (1024 * 1024));
    }
}

// ==================== 执行计划 ====================

void Scheduler::compute_tensor_levels(Tensor* root) {
    if (!root) return;

    std::unordered_set<Tensor*> visited;
    std::queue<Tensor*> queue;

    // 从根节点开始 BFS
    queue.push(root);
    m_tensor_levels[root] = 0;

    while (!queue.empty()) {
        Tensor* current = queue.front();
        queue.pop();

        if (visited.find(current) != visited.end()) continue;
        visited.insert(current);

        int current_level = m_tensor_levels[current];

        // 处理所有输入
        for (auto* src : current->src) {
            if (src) {
                if (m_tensor_levels.find(src) == m_tensor_levels.end()) {
                    m_tensor_levels[src] = current_level - 1;
                } else {
                    m_tensor_levels[src] = std::min(m_tensor_levels[src], current_level - 1);
                }
                queue.push(src);
            }
        }
    }
}

void Scheduler::build_execute_order(Tensor* root) {
    if (!root) {
        std::println("Warning: Root node is null");
        return;
    }

    std::println("Building execute order...");

    m_execute_order.clear();

    // 使用拓扑排序生成执行顺序
    std::unordered_set<Tensor*> executed;
    std::queue<Tensor*> ready_queue;

    // 找到所有没有输入的节点（权重/输入）
    std::unordered_set<Tensor*> has_output;
    for (auto* node : m_graph->m_nodes) {
        for (auto* src : node->src) {
            if (src) {
                has_output.insert(src);
            }
        }
    }

    // 初始就绪节点（所有输入都已就绪）
    for (auto* node : m_graph->m_nodes) {
        bool all_inputs_ready = true;
        for (auto* src : node->src) {
            if (src && std::find(m_graph->m_nodes.begin(), m_graph->m_nodes.end(), src) != m_graph->m_nodes.end()) {
                // src 是计算节点，需要等待
                all_inputs_ready = false;
                break;
            }
        }

        if (all_inputs_ready) {
            ready_queue.push(node);
        }
    }

    int priority = 0;
    while (!ready_queue.empty()) {
        Tensor* current = ready_queue.front();
        ready_queue.pop();

        if (executed.find(current) != executed.end()) continue;

        // 找到当前节点所属的子图
        SubGraph* sg = nullptr;
        for (auto& subgraph : m_subgraphs) {
            if (std::find(subgraph.nodes.begin(), subgraph.nodes.end(), current) != subgraph.nodes.end()) {
                sg = &subgraph;
                break;
            }
        }

        m_execute_order.emplace_back(current, sg, priority++);

        // 标记为已执行
        executed.insert(current);

        // 检查依赖此节点的其他节点是否就绪
        for (auto* node : m_graph->m_nodes) {
            if (executed.find(node) != executed.end()) continue;

            bool all_inputs_executed = true;
            for (auto* src : node->src) {
                if (src && std::find(m_graph->m_nodes.begin(), m_graph->m_nodes.end(), src) != m_graph->m_nodes.end()) {
                    if (executed.find(src) == executed.end()) {
                        all_inputs_executed = false;
                        break;
                    }
                }
            }

            if (all_inputs_executed) {
                ready_queue.push(node);
            }
        }
    }

    std::println("  Built execute order with {} tasks", m_execute_order.size());
}

void Scheduler::optimize_execute_order() {
    // TODO: 实现优化
    // 1. 设备间并行：不同设备上的子图可以并行执行
    // 2. 内存复用：相同大小的临时张量可以复用内存
    // 3. 算子融合：连续的简单算子可以融合为一个复杂算子
    std::println("Optimizing execute order...");
}

// ==================== 调度执行 ====================

void Scheduler::execute(Tensor* root) {
    if (!m_execute_order.empty()) {
        // 使用预构建的执行顺序
        for (const auto& task : m_execute_order) {
            execute_node(task.node);
        }
    } else {
        // 动态构建执行顺序
        build_execute_order(root);
        execute(root);
    }
}

void Scheduler::execute_node(Tensor* node) {
    if (!node) return;

    // TODO: 实际执行算子
    // 这里需要根据 op_type 调用对应的算子实现
    // 例如：
    // switch (node->op_type) {
    //     case OperationType::OP_TYPE_ADD:
    //         kernel_add(node->src[0], node->src[1], node);
    //         break;
    //     case OperationType::OP_TYPE_MUL_MAT:
    //         kernel_mul_mat(node->src[0], node->src[1], node);
    //         break;
    //     ...
    // }

    std::println("Executing: {} ({})", node->name, operation_type_to_string(node->op_type));
}

void Scheduler::cleanup_temp_tensors() {
    // TODO: 释放临时张量的内存
    std::println("Cleaning up temporary tensors...");
}

// ==================== 调试信息 ====================

void Scheduler::print_subgraphs() const {
    std::println("\n=== Subgraphs ({}) ===", m_subgraphs.size());
    for (const auto& sg : m_subgraphs) {
        std::println("SubGraph {}: {} nodes, {} weights",
            sg.id, sg.nodes.size(), sg.weights.size());
        if (sg.device) {
            std::println("  Device: {}:{}",
                static_cast<int>(sg.device->backend),
                sg.device->device_id);
        }
        std::println("  Weight memory: {} MB", sg.weight_memory / (1024 * 1024));
        std::println("  Temp memory: {} MB", sg.temp_memory / (1024 * 1024));
    }
}

void Scheduler::print_execute_order() const {
    std::println("\n=== Execute Order ({}) ===", m_execute_order.size());
    for (size_t i = 0; i < m_execute_order.size(); ++i) {
        const auto& task = m_execute_order[i];
        std::println("[{}] {} on SubGraph {}",
            i, task.node->name, task.subgraph ? task.subgraph->id : -1);
    }
}

void Scheduler::print_memory_stats() const {
    std::println("\n=== Memory Statistics ===");

    size_t total_weight = 0;
    size_t total_temp = 0;

    for (const auto& sg : m_subgraphs) {
        total_weight += sg.weight_memory;
        total_temp = std::max(total_temp, sg.temp_memory);
    }

    std::println("Total weight memory: {} MB", total_weight / (1024 * 1024));
    std::println("Peak temp memory: {} MB", total_temp / (1024 * 1024));
    std::println("Total memory: {} MB", (total_weight + total_temp) / (1024 * 1024));

    // 按设备统计
    std::unordered_map<std::string, size_t> device_weight;
    std::unordered_map<std::string, size_t> device_temp;

    for (const auto& sg : m_subgraphs) {
        if (sg.device) {
            std::string key = std::format("{}:{}",
                static_cast<int>(sg.device->backend),
                sg.device->device_id);
            device_weight[key] += sg.weight_memory;
            device_temp[key] = std::max(device_temp[key], sg.temp_memory);
        }
    }

    std::println("\nPer-device memory:");
    for (const auto& [dev, weight] : device_weight) {
        std::println("  {}: weight={} MB, temp={} MB",
            dev, weight / (1024 * 1024), device_temp[dev] / (1024 * 1024));
    }

    // 打印内存管理器的统计
    if (m_memory_manager) {
        m_memory_manager->print_stats();
    }
}

// ==================== 内存分配 ====================

void Scheduler::allocate_weights_for_subgraph(SubGraph& subgraph) {
    if (!subgraph.device) {
        throw std::runtime_error(std::format("SubGraph {} has no device assigned", subgraph.id));
    }

    std::println("Allocating weights for SubGraph {} on {}...",
        subgraph.id, subgraph.device->to_string());

    for (auto* weight : subgraph.weights) {
        if (!weight || weight->data) {
            continue;  // 跳过空权重或已分配的权重
        }

        try {
            m_memory_manager->allocate_weight(weight, *subgraph.device);
        } catch (const std::exception& e) {
            std::println("  Failed to allocate {}: {}", weight->name, e.what());
            throw;
        }
    }
}

void Scheduler::allocate_all_weights() {
    if (!m_memory_manager) {
        throw std::runtime_error("Memory manager not initialized");
    }

    std::println("\n=== Allocating weights for all subgraphs ===");

    for (auto& subgraph : m_subgraphs) {
        allocate_weights_for_subgraph(subgraph);
    }

    std::println("\nWeight allocation completed");
    m_memory_manager->print_stats();
}

void Scheduler::allocate_temp_memory(Tensor* tensor) {
    if (!tensor || tensor->data) {
        return;
    }

    // TODO: 确定张量应该在哪个设备上分配
    // 简化：暂时分配到 CPU
    const Device* device = m_devices->empty() ? nullptr : &(*m_devices)[0];

    if (device) {
        m_memory_manager->allocate_temp(tensor, *device);
    }
}

void Scheduler::export_schedule_dot(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", filename));
    }

    ofs << "digraph Schedule {\n";
    ofs << "    rankdir=TB;\n";
    ofs << "    node [shape=box];\n\n";

    // 按子图着色
    std::unordered_map<int, std::string> subgraph_colors = {
        {0, "lightblue"},
        {1, "lightgreen"},
        {2, "lightyellow"},
        {3, "lightpink"},
    };

    // 导出节点
    for (const auto& task : m_execute_order) {
        std::string node_id = std::format("node_{}", reinterpret_cast<uintptr_t>(task.node));
        std::string color = task.subgraph ?
            subgraph_colors[task.subgraph->id % subgraph_colors.size()] : "white";

        ofs << "    \"" << node_id << "\" [label=\""
            << task.node->name << "\\n"
            << operation_type_to_string(task.node->op_type)
            << "\", fillcolor=" << color << ", style=filled];\n";

        // 导出边
        for (auto* src : task.node->src) {
            if (src) {
                std::string src_id = std::format("node_{}", reinterpret_cast<uintptr_t>(src));
                ofs << "    \"" << src_id << "\" -> \"" << node_id << "\";\n";
            }
        }
    }

    ofs << "}\n";
    ofs.close();
}
