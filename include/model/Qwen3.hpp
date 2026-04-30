#pragma once
#include <cstddef>

#include "model/model.h"

// Qwen3 模型结构参数（从 GGUF metadata 解析）
struct Qwen3Config {
    int hidden_size = 1024;
    int num_layers = 28;
    int num_heads = 16;
    int num_kv_heads = 8;
    int head_dim = 128;
    int intermediate_size = 3072;
    int vocab_size = 0;
    int max_seq_len = 40960;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

// Qwen3 模型类
class Qwen3Model : public ModelBase {
private:
    Qwen3Config config_;
    void parse_config(const GGUFInfo& info);
    [[nodiscard]] Tensor* build_qwen3_layer(
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

    ~Qwen3Model(){
        // 没有什么可以清理的
    }

    // 禁止拷贝，允许移动
    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;
    Qwen3Model(Qwen3Model&&) noexcept = default;
    Qwen3Model& operator=(Qwen3Model&&) noexcept = default;

    void print_info() override;
    ComputeGraph& build_graph(const GGUFInfo& info) override;
    [[nodiscard]] const Qwen3Config& config() const { return config_; }
    [[nodiscard]]size_t vocab_size() const override{ return config_.vocab_size; }

    [[nodiscard]] int64_t max_seq_len() const override { return config_.max_seq_len; }

};
