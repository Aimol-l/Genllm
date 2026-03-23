// executor.cpp - 执行器实现
#include "core/executor.h"
#include "core/operator.h"
#include "cpu/cpu_ops.h"
#include "cuda/cuda_ops.h"
#include <chrono>
#include <thread>

// ==================== 执行控制 ====================

void Executor::execute(Tensor* root) {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::println("\n=== Starting graph execution ===");

    // 从调度器获取执行顺序
    const auto& execute_order = m_scheduler->get_execute_order();

    if (execute_order.empty()) {
        std::println("Warning: Execute order is empty, building from root...");
        m_scheduler->build_execute_order(root);
    }

    // 构建执行节点
    build_execution_nodes();

    // 打印执行计划
    print_execution_plan();

    // 初始化就绪队列
    update_ready_queue();

    // 根据配置选择执行模式
    // TODO: 根据设备数量和配置决定是否并行
    execute_sequential(root);

    auto end_time = std::chrono::high_resolution_clock::now();
    m_total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();

    std::println("\n=== Execution completed in {} ms ===", m_total_time_ms);
    print_statistics();
}

void Executor::execute_sequential(Tensor* root) {
    std::println("\n--- Sequential execution mode ---");

    const auto& execute_order = m_scheduler->get_execute_order();

    for (size_t i = 0; i < execute_order.size(); ++i) {
        const auto& task = execute_order[i];

        // 打印执行信息
        std::println("[{}/{}] Executing: {} on {}",
            i + 1, execute_order.size(),
            task.node ? task.node->name : "null",
            task.subgraph ? task.subgraph->device_str() : "no device");

        // 执行节点
        dispatch_operation(task.node);

        // 标记为已完成
        m_completed_tensors.insert(task.node);
        m_total_executed++;

        // 清理不再需要的临时张量
        cleanup_temp_tensor(task.node);
    }
}

void Executor::execute_parallel(Tensor* root) {
    std::println("\n--- Parallel execution mode ---");

    // TODO: 实现多线程并行执行
    // 1. 为每个设备启动一个工作线程
    // 2. 就绪队列中的节点可以并行执行（如果在不同设备上）
    // 3. 节点完成后，检查并激活依赖它的节点

    throw std::runtime_error("Parallel execution not implemented yet");
}

void Executor::execute_node(ExecutionNode* node) {
    if (!node || !node->tensor) {
        return;
    }

    std::println("Executing node: {} on device: {}",
        node->tensor->name,
        node->subgraph ? node->subgraph->device_str() : "none");

    node->status = ExecutionStatus::RUNNING;

    // 执行算子
    dispatch_operation(node->tensor);

    node->status = ExecutionStatus::COMPLETED;
    m_total_executed++;

    // 激活依赖此节点的其他节点
    for (auto* dependent : node->dependents) {
        if (dependent->is_ready()) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_ready_queue.push(dependent);
            m_cv.notify_one();
        }
    }
}

// ==================== 节点管理 ====================

void Executor::build_execution_nodes() {
    const auto& execute_order = m_scheduler->get_execute_order();

    // 创建执行节点
    for (const auto& task : execute_order) {
        auto node = std::make_unique<ExecutionNode>(
            task.node, task.subgraph, task.priority);
        m_nodes.push_back(std::move(node));
    }

    // 构建依赖关系
    for (auto& node : m_nodes) {
        if (!node->tensor) continue;

        // 检查所有输入（src）
        for (auto* src : node->tensor->src) {
            if (!src) continue;

            // 找到对应的执行节点
            for (auto& other : m_nodes) {
                if (other->tensor == src) {
                    node->dependencies.push_back(other.get());
                    other->dependents.push_back(node.get());
                    break;
                }
            }
        }
    }
}

void Executor::update_ready_queue() {
    for (auto& node : m_nodes) {
        if (node->is_ready()) {
            m_ready_queue.push(node.get());
        }
    }

    std::println("Initial ready queue size: {}", m_ready_queue.size());
}

// ==================== 算子执行 ====================

void Executor::dispatch_operation(Tensor* tensor) {
    if (!tensor) return;

    // 跳过叶子节点（权重或输入）
    if (tensor->op_type == OperationType::OP_TYPE_NONE) {
        return;
    }

    std::println("  Dispatching: {} ({})",
        tensor->name,
        operation_type_to_string(tensor->op_type));

    try {
        // 使用算子分发器自动选择对应设备的内核
        OpDispatcher& dispatcher = get_op_dispatcher();

        // 创建执行上下文
        OpContext ctx;
        // TODO: 根据 tensor->backend 设置 ctx.device
        // TODO: 创建 CUDA stream（如果需要）

        // 分发并执行算子
        // dispatcher 会根据 tensor->backend 自动选择 CPU 或 CUDA 内核
        dispatcher.dispatch(tensor, ctx);

        std::println("    ✓ Executed successfully");

    } catch (const std::exception& e) {
        std::println("    ✗ Execution failed: {}", e.what());
        throw;
    }
}

void Executor::cleanup_temp_tensor(Tensor* tensor) {
    if (!tensor) return;

    // TODO: 释放不再需要的临时张量内存
    // 规则：
    // 1. 叶子节点（权重）不释放
    // 2. 如果这个张量是其他节点的输入，不能立即释放
    // 3. 只有在所有依赖它的节点都执行完成后，才能释放

    // 简化实现：暂时不释放，等整个推理完成后统一清理
}

// ==================== 调试和统计 ====================

void Executor::print_execution_plan() const {
    std::println("\n=== Execution Plan ===");
    std::println("Total nodes: {}", m_nodes.size());

    // 按优先级（拓扑层级）分组
    std::unordered_map<int, std::vector<const ExecutionNode*>> level_groups;
    for (const auto& node : m_nodes) {
        level_groups[node->priority].push_back(node.get());
    }

    std::println("\nExecution order by level:");
    for (const auto& [level, nodes] : level_groups) {
        std::println("  Level {}: {} nodes", level, nodes.size());
        for (const auto* node : nodes) {
            std::println("    - {} on {} ({})",
                node->tensor ? node->tensor->name : "null",
                node->subgraph ? node->subgraph->device_str() : "no device",
                operation_type_to_string(node->tensor->op_type));
        }
    }
}

void Executor::print_statistics() const {
    std::println("\n=== Execution Statistics ===");
    std::println("Total nodes executed: {}", m_total_executed);
    std::println("Total time: {} ms", m_total_time_ms);

    if (m_total_time_ms > 0) {
        double throughput = static_cast<double>(m_total_executed) /
                           (m_total_time_ms / 1000.0);
        std::println("Throughput: {:.2f} nodes/sec", throughput);
    }

    // 按设备统计
    std::unordered_map<std::string, size_t> device_exec_count;
    for (const auto& node : m_nodes) {
        if (node->subgraph) {
            std::string device_key = node->subgraph->device_str();
            device_exec_count[device_key]++;
        }
    }

    if (!device_exec_count.empty()) {
        std::println("\nPer-device execution count:");
        for (const auto& [device, count] : device_exec_count) {
            std::println("  {}: {} nodes", device, count);
        }
    }
}

// ==================== 工作线程（并行模式） ====================

void Executor::worker_thread() {
    while (true) {
        std::unique_lock<std::mutex> lock(m_mutex);

        // 等待就绪节点或停止信号
        m_cv.wait(lock, [this] {
            return !m_ready_queue.empty() || m_stop_flag;
        });

        if (m_stop_flag && m_ready_queue.empty()) {
            break;
        }

        if (!m_ready_queue.empty()) {
            ExecutionNode* node = m_ready_queue.front();
            m_ready_queue.pop();

            lock.unlock();

            // 执行节点
            execute_node(node);
        }
    }
}
