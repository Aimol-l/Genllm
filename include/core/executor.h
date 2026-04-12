#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "core/scheduler.h"

class Executor {
public:
    explicit Executor(GraphScheduler& scheduler);

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    /// 自回归生成：prefill(prompt) → 循环 decode → 返回生成的 token 序列
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt,
        int max_tokens,
        int eos_token = -1
    );

private:
    GraphScheduler& scheduler_;
    MemoryManager& memory_;
    const ComputeGraph& graph_;
    std::unordered_map<Device, size_t> dev_id_map_;

    struct InputBinding { void* data; size_t size; };
    std::unordered_map<std::string, InputBinding> inputs_;

    // ========== 内部状态 ==========
    bool is_prefill_ = true;
    int64_t seq_pos_ = 0;           // 当前序列位置（KV cache 索引）

    // ========== prefill / decode ==========
    /// 处理整个 prompt（seq_len = prompt.size()）
    void prefill(const std::vector<int32_t>& token_ids);

    /// 处理单个新 token（seq_len = 1），返回采样结果
    int32_t decode_step(int32_t token);

    // ========== 底层执行 ==========
    /// 执行一次前向传播
    void forward();

    /// 从 logits 输出采样下一个 token（stub，后续实现）
    int32_t sample() const;

    void reset_activations();
    void resolve_dims(int64_t batch, int64_t seq_len);
    void bind_input(const std::string& name, void* data, size_t byte_size);

    void allocate_output(Tensor* t);
    void execute_view(Tensor* t);
    void execute_memcpy(Tensor* t);
    void dispatch_kernel(Tensor* t);
    MemoryPool* get_act_pool(Device dev) const;

    Tensor* find_tensor(const std::string& name) const;

    static bool is_view_op(OperationType op) {
        return op == OperationType::OP_TYPE_RESHAPE ||
               op == OperationType::OP_TYPE_PERMUTE  ||
               op == OperationType::OP_TYPE_VIEW     ||
               op == OperationType::OP_TYPE_TRANSPOSE;
    }
};
