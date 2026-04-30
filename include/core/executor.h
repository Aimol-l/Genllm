#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "scheduler.h"
#include "thread_pool.h"

class Executor {
private:
    struct LayerGroup {
        int32_t layer_id  = 0;
        int32_t device_id = 0;
        std::vector<std::vector<Tensor*>> levels; // 层内的依赖子级别
    };

    bool is_prefill_          = true;   // 是否预填充
    bool persistent_computed_ = false;
    int64_t seq_pos_          = 0;      // 当前序列位置（KV cache 索引）

    GraphScheduler& scheduler_;
    MemoryManager& memory_;
    const ComputeGraph& graph_;
    std::map<Device, std::vector<int>> dev_id_map_;
    struct InputBinding { void* data; size_t size; };
    std::map<std::string, InputBinding> inputs_;
    std::unique_ptr<ThreadPool> pool_;  // 固定线程池，生命周期跟随 Executor
    
    std::vector<Tensor*> apply_rope_tensors_;
    std::vector<LayerGroup> step_layers_;        // transformer 层，按 layer_id 升序
    std::vector<LayerGroup> persistent_layers_;  // CACHE 类型 (layer_id < 0)
    
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
        int32_t eos_tokens,
        class Tokenizer* tokenizer = nullptr
    );

    [[nodiscard]] int64_t seq_pos() const { return seq_pos_; }

    /// 多轮对话：在已有 KV cache 基础上追加新 token 并做 prefill
    void append_tokens(const std::vector<int32_t>& token_ids);
    void decode_step(int32_t token_id);
    [[nodiscard]] int32_t sample() const;

private:
    void prefill(const std::vector<int32_t>& token_ids);

    void forward();
    [[nodiscard]] int32_t sample_argmax() const;  // greedy：直接取 argmax
    [[nodiscard]] int32_t sample_top_p(float temperature, float top_p) const;
    void reset_activations();
    void resolve_dims(int64_t batch, int64_t seq_len);
    void bind_input(const std::string& name, void* data, size_t byte_size);
    void execute_view(Tensor* t);
    void allocate_output(Tensor* t,int32_t dev_id = 0);
    void execute_tensor(Tensor* t,int32_t dev_id = 0);
    void dispatch_kernel(Tensor* t,int32_t dev_id = 0);
    void reset_step_activations();
    [[nodiscard]] MemoryPool* get_act_pool(Device dev,int32_t dev_id = 0) const;
    [[nodiscard]] Tensor* find_tensor(const std::string& name) const;
    static bool is_view_op(OperationType op) {
        return op == OperationType::OP_TYPE_RESHAPE ||
               op == OperationType::OP_TYPE_VIEW     ||
               op == OperationType::OP_TYPE_TRANSPOSE;
    }
};
