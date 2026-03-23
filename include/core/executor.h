// executor.h - 计算图执行器
#pragma once
#include <vector>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <print>
#include <functional>
#include "scheduler.h"
#include "tensor.hpp"

// 执行状态
enum class ExecutionStatus {
    PENDING,    // 等待执行
    READY,      // 就绪（输入都已准备好）
    RUNNING,    // 正在执行
    COMPLETED,  // 已完成
    FAILED      // 执行失败
};

// 执行节点（扩展 ExecuteTask）
struct ExecutionNode {
    Tensor* tensor;                  // 要执行的张量节点
    SubGraph* subgraph;              // 所属子图（决定设备）
    int priority;                    // 拓扑优先级
    ExecutionStatus status;          // 执行状态
    std::vector<ExecutionNode*> dependencies;  // 依赖的节点
    std::vector<ExecutionNode*> dependents;    // 依赖此节点的节点

    ExecutionNode(Tensor* t, SubGraph* sg, int p)
        : tensor(t), subgraph(sg), priority(p), status(ExecutionStatus::PENDING) {}

    // 检查是否就绪（所有依赖都已完成）
    [[nodiscard]] bool is_ready() const {
        if (status != ExecutionStatus::PENDING) {
            return false;
        }
        for (const auto* dep : dependencies) {
            if (dep->status != ExecutionStatus::COMPLETED) {
                return false;
            }
        }
        return true;
    }

    // 检查是否可以并行（在不同设备上）
    [[nodiscard]] bool can_run_parallel_with(const ExecutionNode* other) const {
        if (!subgraph || !other->subgraph) return false;
        return subgraph->device != other->subgraph->device;
    }
};

// 执行器类
class Executor {
private:
    Scheduler* m_scheduler;
    std::vector<std::unique_ptr<ExecutionNode>> m_nodes;
    std::queue<ExecutionNode*> m_ready_queue;
    std::unordered_set<Tensor*> m_completed_tensors;

    // 并发执行控制
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stop_flag;

    // 统计信息
    size_t m_total_executed;
    size_t m_total_time_ms;

public:
    explicit Executor(Scheduler* scheduler)
        : m_scheduler(scheduler)
        , m_stop_flag(false)
        , m_total_executed(0)
        , m_total_time_ms(0) {}

    ~Executor() = default;

    // ==================== 执行控制 ====================

    // 执行整个计算图（从根节点开始）
    void execute(Tensor* root);

    // 执行单个节点
    void execute_node(ExecutionNode* node);

    // 单线程顺序执行（调试用）
    void execute_sequential(Tensor* root);

    // 多线程并行执行（利用多设备）
    void execute_parallel(Tensor* root);

    // 停止执行
    void stop() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop_flag = true;
        m_cv.notify_all();
    }

    // ==================== 节点管理 ====================

    // 从调度器的执行计划构建执行节点
    void build_execution_nodes();

    // 更新就绪队列
    void update_ready_queue();

    // ==================== 算子执行 ====================

    // 执行具体的算子操作
    void dispatch_operation(Tensor* tensor);

    // ==================== 调试和统计 ====================

    // 打印执行计划
    void print_execution_plan() const;

    // 打印执行统计
    void print_statistics() const;

    // 获取执行统计
    [[nodiscard]] size_t get_total_executed() const { return m_total_executed; }
    [[nodiscard]] size_t get_total_time_ms() const { return m_total_time_ms; }

private:
    // 执行工作线程（并行模式）
    void worker_thread();

    // 清理临时张量
    void cleanup_temp_tensor(Tensor* tensor);
};

// ==================== 执行流程说明 ====================
/*
 *
 * 执行顺序示例：
 *
 * 假设计算图如下：
 *
 *   input_ids (CPU)
 *       ↓
 *   embd_lookup (CPU) → embd [子图0:CPU]
 *       ↓
 *   attn_norm (CPU) → normed [子图0:CPU]
 *       ↓
 *   split into:
 *     - Q projection (GPU0) → Q [子图1:GPU0]
 *     - K projection (GPU1) → K [子图2:GPU1]
 *     - V projection (GPU1) → V [子图2:GPU1]
 *       ↓
 *   attention (GPU0) → attn_out [子图1:GPU0]
 *       ↓
 *   ffn (GPU1) → ffn_out [子图2:GPU1]
 *       ↓
 *   add (CPU) → output [子图0:CPU]
 *
 *
 * 执行顺序（拓扑序）：
 * 1. input_ids      (CPU)   ← 独立执行
 * 2. embd_lookup    (CPU)   ← 等待 1
 * 3. attn_norm      (CPU)   ← 等待 2
 * 4. Q_projection   (GPU0)  ← 等待 3，可与 5,6 并行
 * 5. K_projection   (GPU1)  ← 等待 3，可与 4,6 并行
 * 6. V_projection   (GPU1)  ← 等待 3，可与 4,5 并行
 * 7. attention      (GPU0)  ← 等待 4,5,6
 * 8. ffn            (GPU1)  ← 等待 7
 * 9. add            (CPU)   ← 等待 8
 *
 *
 * 关键点：
 * - 按拓扑序执行，不是按子图顺序
 * - 不同设备上无依赖的节点可以并行执行（如 4,5,6）
 * - 节点在哪执行由子图决定（device）
 * - 内存分配也由子图决定（memory pool）
 *
 */
