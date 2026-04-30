#pragma once
#include <cstddef>

#include "model/model.h"

// Qwen35 模型结构参数（从 GGUF metadata 解析）
struct Qwen35Config {
    int block_count = 24;
    int context_length = 262144;
    int embedding_length = 1024;
    int feed_forward_length = 3584;
    int head_count = 8;
    int head_count_kv = 2;
    int key_length = 256;
    int value_length = 256;
    int rope_dimension_count = 64;
    int ssm_conv_kernel = 4;
    int ssm_state_size = 128;
    int ssm_group_count = 16;
    int ssm_time_step_rank = 16;
    int ssm_inner_size = 2048;
    int full_attention_interval = 4;
    int vocab_size = 0;
    int max_seq_len = 262144;
    float rope_theta = 10000000.0f;
    float rms_norm_eps = 1e-6f;
    std::vector<int> rope_dimension_sections;
};

// Qwen3 模型类
class Qwen35Model : public ModelBase {
private:
    Qwen35Config config_;
    void parse_config(const GGUFInfo& info);
    Tensor* build_ssm_layer(Tensor* input, const GGUFInfo& info, int layer_idx);
    Tensor* build_attn_layer(Tensor* input, const GGUFInfo& info, int layer_idx, Tensor* rope_cos, Tensor* rope_sin);
public:
    Qwen35Model() {
        name = "Qwen35";
        type = ModelType::CAUSAL_LM;
        arch = ModelArch::QWEN35;
    }

    ~Qwen35Model(){
        // 没有什么可以清理的
    }

    // 禁止拷贝，允许移动
    Qwen35Model(const Qwen35Model&) = delete;
    Qwen35Model& operator=(const Qwen35Model&) = delete;
    Qwen35Model(Qwen35Model&&) noexcept = default;
    Qwen35Model& operator=(Qwen35Model&&) noexcept = default;

    void print_info() override;
    ComputeGraph& build_graph(const GGUFInfo& info) override;
    [[nodiscard]] const Qwen35Config& config() const { return config_; }
    [[nodiscard]]size_t vocab_size() const override{ return config_.vocab_size; }

    [[nodiscard]] int64_t max_seq_len() const override { return config_.max_seq_len; }

};
