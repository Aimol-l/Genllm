#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "scheduler.h"
#include "thread_pool.h"

class Executor {
private:
 // ========== 内部状态 ==========
    bool is_prefill_ = true;
    bool persistent_computed_ = false;
    int64_t seq_pos_ = 0;           // 当前序列位置（KV cache 索引）

    GraphScheduler& scheduler_;
    MemoryManager& memory_;
    const ComputeGraph& graph_;
    std::unordered_map<Device, size_t> dev_id_map_;
    struct InputBinding { void* data; size_t size; };
    std::unordered_map<std::string, InputBinding> inputs_;

    std::unique_ptr<ThreadPool> pool_;  // 固定线程池，生命周期跟随 Executor

    struct LayerGroup {
        int layer_id;
        std::vector<std::vector<Tensor*>> levels; // 层内的依赖子级别
    };
    std::vector<LayerGroup> persistent_layers_;  // CACHE 类型 (layer_id < 0)
    std::vector<LayerGroup> step_layers_;        // transformer 层，按 layer_id 升序
    
    std::unordered_map<Device, size_t> persistent_cursor_;
    std::unordered_map<std::string, std::array<int64_t, TENSOR_MAX_DIMS>> original_dims_;
    
public:
    explicit Executor(GraphScheduler& scheduler);
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    /// 自回归生成：prefill(prompt) → 循环 decode → 返回生成的 token 序列
    std::vector<int32_t> generate(
        const std::vector<int32_t>& prompt,
        int64_t max_tokens,
        int32_t eos_tokens
    );
private:
    void prefill(const std::vector<int32_t>& token_ids);
    void decode_step(const std::vector<int32_t>& token_ids);

    void forward();
    int32_t sample() const;
    void reset_activations();
    void resolve_dims(int64_t batch, int64_t seq_len);
    void bind_input(const std::string& name, void* data, size_t byte_size);
    void allocate_output(Tensor* t);
    void execute_view(Tensor* t);
    void execute_memcpy(Tensor* t);
    void execute_tensor(Tensor* t);
    void dispatch_kernel(Tensor* t);
    void reset_step_activations();
    MemoryPool* get_act_pool(Device dev) const;
    Tensor* find_tensor(const std::string& name) const;
    static bool is_view_op(OperationType op) {
        return op == OperationType::OP_TYPE_RESHAPE ||
               op == OperationType::OP_TYPE_VIEW     ||
               op == OperationType::OP_TYPE_TRANSPOSE;
    }
};
