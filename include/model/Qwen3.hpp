#pragma once

#include "model.h"


struct Qwen3Params{
    float padding = 0.2;
};

// Qwen3 模型类
class Qwen3Model : public ModelBase {
private:
    Qwen3Params params;
    Tensor* build_qwen3_layer(
        Tensor* input,           // [batch, seq_len, hidden_size]
        const GGUFInfo& info,
        int layer_idx,
        int hidden_size,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int intermediate_size,
        float rms_norm_eps,
        Tensor* rope_cos,
        Tensor* rope_sin
    );
public:
    Qwen3Model() {
        name = "Qwen3";
        type = ModelType::CAUSAL_LM;
        arch = ModelArch::QWEN3;
    }

    ~Qwen3Model() override = default;

    // 禁止拷贝，允许移动
    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;
    Qwen3Model(Qwen3Model&&) noexcept = default;
    Qwen3Model& operator=(Qwen3Model&&) noexcept = default;

    void print_info() override;
    void set_params(void*) override;
    ComputeGraph* build_graph(const GGUFInfo& info) override;

    // 加载权重（使用内存管理器）
    void load_weights(GGUFInfo& info, MemoryManager* mem_manager) override;
};
