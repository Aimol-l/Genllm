#include "model/Qwen35.hpp"
#include "model/op_factory.hpp"
#include "utils/tools.hpp"

void Qwen35Model::print_info(){
    std::println("=== Qwen3.5 Hybrid Model Info ===");
    std::println("  Name: {}", name);
    std::println("  block_count:         {}", config_.block_count);
    std::println("  context_length:      {}", config_.context_length);
    std::println("  embedding_length:    {}", config_.embedding_length);
    std::println("  feed_forward_length: {}", config_.feed_forward_length);
    std::println("  head_count:          {}", config_.head_count);
    std::println("  head_count_kv:       {}", config_.head_count_kv);
    std::println("  key_length:          {}", config_.key_length);
    std::println("  value_length:        {}", config_.value_length);
    std::println("  ssm_conv_kernel:     {}", config_.ssm_conv_kernel);
    std::println("  rope_dimension_count:{}", config_.rope_dimension_count);
    std::println("  ssm_state_size:      {}", config_.ssm_state_size);
    std::println("  ssm_group_count:     {}", config_.ssm_group_count);
    std::println("  ssm_time_step_rank:  {}", config_.ssm_time_step_rank);
    std::println("  ssm_inner_size:      {}", config_.ssm_inner_size);
    std::println("  full_attention_interval: {}", config_.full_attention_interval);
    std::println("  vocab_size:          {}", config_.vocab_size);
    std::println("  max_seq_len:         {}", config_.max_seq_len);
    std::println("  rope_theta:          {}", config_.rope_theta);
    std::println("  rms_norm_eps:        {}", config_.rms_norm_eps);
    std::println("  rope_dimension_sections:{}", config_.rope_dimension_sections);
}

void Qwen35Model::parse_config(const GGUFInfo& info) {
    auto& meta = info.metadata;
    constexpr std::string_view prefix = "qwen35.";

    config_.block_count         = meta.value(std::string(prefix) + "block_count", 24);
    config_.context_length      = meta.value(std::string(prefix) + "context_length", 262144);
    config_.max_seq_len         = config_.context_length;
    config_.embedding_length    = meta.value(std::string(prefix) + "embedding_length", 1024);
    config_.feed_forward_length = meta.value(std::string(prefix) + "feed_forward_length", 3584);

    config_.head_count          = meta.value(std::string(prefix) + "attention.head_count", 8);
    config_.head_count_kv       = meta.value(std::string(prefix) + "attention.head_count_kv", 2);
    config_.key_length          = meta.value(std::string(prefix) + "attention.key_length", 256);
    config_.value_length        = meta.value(std::string(prefix) + "attention.value_length", 256);
    config_.rms_norm_eps        = meta.value(std::string(prefix) + "attention.layer_norm_rms_epsilon", 1e-6f);

    config_.rope_theta          = meta.value(std::string(prefix) + "rope.freq_base", 10000000.0f);
    config_.rope_dimension_count= meta.value(std::string(prefix) + "rope.dimension_count", 64);

    config_.ssm_conv_kernel     = meta.value(std::string(prefix) + "ssm.conv_kernel", 4);
    config_.ssm_state_size      = meta.value(std::string(prefix) + "ssm.state_size", 128);
    config_.ssm_group_count     = meta.value(std::string(prefix) + "ssm.group_count", 16);
    config_.ssm_time_step_rank  = meta.value(std::string(prefix) + "ssm.time_step_rank", 16);
    config_.ssm_inner_size      = meta.value(std::string(prefix) + "ssm.inner_size", 2048);

    config_.full_attention_interval = meta.value(std::string(prefix) + "full_attention_interval", 4);

    std::string sec_key = std::string(prefix) + "rope.dimension_sections";
    if (meta.contains(sec_key)) {
        const auto& sec = meta[sec_key];
        if (sec.is_array()) {
            config_.rope_dimension_sections = sec.get<std::vector<int>>();
        }
    }

    if (meta.contains("tokenizer.ggml.tokens") && meta["tokenizer.ggml.tokens"].is_array()) {
        config_.vocab_size = static_cast<int>(meta["tokenizer.ggml.tokens"].size());
    }
    if (config_.vocab_size == 0) {
        for (const auto& t : info.tensors_info) {
            if (t.name == "token_embd.weight" && t.dimensions.size() == 2) {
                config_.vocab_size = static_cast<int>(t.dimensions[1]);
                break;
            }
        }
    }
    if (config_.vocab_size == 0) {
        throw std::runtime_error("Cannot determine vocab_size from GGUF metadata or tensors.");
    }

    std::println("Qwen3.5 Config parsed:");
    std::println("  layers={}, hidden={}, ffn={}, heads={}/kv={}",
        config_.block_count, config_.embedding_length, config_.feed_forward_length,
        config_.head_count, config_.head_count_kv);
    std::println("  ssm_inner={}, ssm_state={}, attn_interval={}, rope_theta={}",
        config_.ssm_inner_size, config_.ssm_state_size, config_.full_attention_interval, config_.rope_theta);
}

ComputeGraph& Qwen35Model::build_graph(const GGUFInfo& info){
    std::println("Building Qwen3.5 Hybrid (Mamba2+Attn) computation graph...");
    this->parse_config(info);

    Tensor* input_ids = OpFactory::placeholder(DataType::GGML_TYPE_I32, TensorType::TENSOR_TYPE_INPUT, {1, -1}, "input_ids");
    
    auto [rope_cos, rope_sin] = OpFactory::rope_cache(
        config_.max_seq_len, 
        config_.rope_dimension_count,
        config_.rope_theta, 
        DataType::GGML_TYPE_F32
    );

    const TensorInfo* embd_weight_info = OpFactory::find_tensor(info, "token_embd.weight");
    if (!embd_weight_info)  throw std::runtime_error("token_embd.weight not found");

    Tensor* x_in = OpFactory::embedding_lookup(input_ids, embd_weight_info, "x_in", -1, false);
    Tensor* prev_output = x_in;

    for (int i = 0; i < config_.block_count; ++i) {
        // 动态路由：每 full_attention_interval 层插入 1 个 Attention
        bool is_attn_layer = ((i + 1) % config_.full_attention_interval == 0);
        if(is_attn_layer){
            prev_output = this->build_attn_layer(prev_output, info, i, rope_cos, rope_sin);
        } else {
            prev_output = this->build_ssm_layer(prev_output, info, i);
        }
    }

    const TensorInfo* output_norm_info = OpFactory::find_tensor(info, "output_norm.weight");

    Tensor* final_norm = OpFactory::rms_norm(prev_output, output_norm_info, config_.rms_norm_eps, "final_norm", config_.block_count);
    
    Tensor* logits = OpFactory::linear(final_norm, embd_weight_info, "logits", config_.block_count, true);
    logits->type = TensorType::TENSOR_TYPE_OUTPUT;
    
    this->graph_.build_from_outputs({logits});
    return this->graph_;
}

Tensor* Qwen35Model::build_ssm_layer(Tensor* input, const GGUFInfo& info, int layer_idx) {
    std::string p = std::format("blk.{}", layer_idx);
    
    auto* norm_w = OpFactory::find_tensor(info, p + ".attn_norm.weight");
    auto* x_norm = OpFactory::rms_norm(input, norm_w, config_.rms_norm_eps, "ssm_x_norm", layer_idx);

    auto* qkv_w = OpFactory::find_tensor(info, p + ".attn_qkv.weight");
    auto* gate_w = OpFactory::find_tensor(info, p + ".attn_gate.weight");
    auto* qkv_proj = OpFactory::linear(x_norm, qkv_w, "ssm_qkv", layer_idx, true);
    auto* z_gate   = OpFactory::linear(x_norm, gate_w, "ssm_z_gate", layer_idx, true);

    auto* conv_w = OpFactory::find_tensor(info, p + ".ssm_conv1d.weight");
    auto* conv_out = OpFactory::causal_conv1d(qkv_proj, conv_w, config_.ssm_conv_kernel, "conv_out", layer_idx);

    auto* ssm_norm_w = OpFactory::find_tensor(info, p + ".ssm_norm.weight");
    auto* ssm_normed = OpFactory::rms_norm(conv_out, ssm_norm_w, config_.rms_norm_eps, "ssm_normed", layer_idx);

    auto* ssm_a     = OpFactory::find_tensor(info, p + ".ssm_a");
    auto* ssm_alpha = OpFactory::find_tensor(info, p + ".ssm_alpha.weight");
    auto* ssm_beta  = OpFactory::find_tensor(info, p + ".ssm_beta.weight");
    auto* ssm_dt    = OpFactory::find_tensor(info, p + ".ssm_dt.bias");
    auto* ssm_out = OpFactory::ssm_scan(ssm_normed, ssm_a, ssm_alpha, ssm_beta, ssm_dt, "ssm_out", layer_idx);

    auto* z_act = OpFactory::silu(z_gate, "z_act", layer_idx);
    auto* mixed = OpFactory::mul(ssm_out, z_act, "ssm_mixed", layer_idx);
    auto* out_w = OpFactory::find_tensor(info, p + ".ssm_out.weight");
    auto* ssm_final = OpFactory::linear(mixed, out_w, "ssm_final", layer_idx, true);

    auto* attn_res = OpFactory::add(ssm_final, input, "ssm_res", layer_idx);
    auto* ffn_norm_w = OpFactory::find_tensor(info, p + ".post_attention_norm.weight");
    auto* ffn_normed = OpFactory::rms_norm(attn_res, ffn_norm_w, config_.rms_norm_eps, "ffn_normed", layer_idx);

    auto* ffn_gate_w = OpFactory::find_tensor(info, p + ".ffn_gate.weight");
    auto* ffn_up_w   = OpFactory::find_tensor(info, p + ".ffn_up.weight");
    auto* ffn_gate = OpFactory::linear(ffn_normed, ffn_gate_w, "ffn_gate", layer_idx, true);
    auto* ffn_up   = OpFactory::linear(ffn_normed, ffn_up_w,   "ffn_up",   layer_idx, true);
    auto* ffn_inter = OpFactory::mul(OpFactory::silu(ffn_gate, "ffn_gate_act", layer_idx), ffn_up, "ffn_inter", layer_idx);

    auto* ffn_down_w = OpFactory::find_tensor(info, p + ".ffn_down.weight");
    auto* ffn_out = OpFactory::linear(ffn_inter, ffn_down_w, "ffn_out", layer_idx, true);
    
    return OpFactory::add(ffn_out, attn_res, "layer_out", layer_idx);
}

Tensor* Qwen35Model::build_attn_layer(Tensor* input, const GGUFInfo& info, int layer_idx, Tensor* rope_cos, Tensor* rope_sin) {
    std::string p = std::format("blk.{}", layer_idx);
    
    auto* norm_w = OpFactory::find_tensor(info, p + ".attn_norm.weight");
    auto* x_norm = OpFactory::rms_norm(input, norm_w, config_.rms_norm_eps, "attn_x_norm", layer_idx);

    auto* q_w = OpFactory::find_tensor(info, p + ".attn_q.weight");
    auto* k_w = OpFactory::find_tensor(info, p + ".attn_k.weight");
    auto* v_w = OpFactory::find_tensor(info, p + ".attn_v.weight");
    auto* q_flat = OpFactory::linear(x_norm, q_w, "q_flat", layer_idx, true);
    auto* k_flat = OpFactory::linear(x_norm, k_w, "k_flat", layer_idx, true);
    auto* v_flat = OpFactory::linear(x_norm, v_w, "v_flat", layer_idx, true);

    auto* q_4d = OpFactory::reshape_permute(q_flat, {1, -1, config_.head_count,    config_.key_length},   {0, 2, 1, 3}, "q_4d", layer_idx);
    auto* k_4d = OpFactory::reshape_permute(k_flat, {1, -1, config_.head_count_kv, config_.key_length},   {0, 2, 1, 3}, "k_4d", layer_idx);
    auto* v_4d = OpFactory::reshape_permute(v_flat, {1, -1, config_.head_count_kv, config_.value_length}, {0, 2, 1, 3}, "v_4d", layer_idx);

    auto* q_norm_w = OpFactory::find_tensor(info, p + ".attn_q_norm.weight");
    auto* k_norm_w = OpFactory::find_tensor(info, p + ".attn_k_norm.weight");
    auto* q_normed = OpFactory::rms_norm(q_4d, q_norm_w, config_.rms_norm_eps, "q_normed", layer_idx);
    auto* k_normed = OpFactory::rms_norm(k_4d, k_norm_w, config_.rms_norm_eps, "k_normed", layer_idx);

    auto [q_rope, k_rope] = OpFactory::apply_rope(q_normed, k_normed, rope_cos, rope_sin, nullptr, "", layer_idx);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(config_.key_length));

    auto* attn_4d = OpFactory::SDPA(q_rope, k_rope, v_4d, nullptr, scale, true, config_.head_count / config_.head_count_kv, "attn_4d", layer_idx);

    auto* attn_flat = OpFactory::permute_reshape(attn_4d, {0, 2, 1, 3}, {1, -1, config_.head_count * config_.key_length}, "attn_flat", layer_idx);
    auto* out_w = OpFactory::find_tensor(info, p + ".attn_output.weight");
    auto* attn_out = OpFactory::linear(attn_flat, out_w, "attn_out", layer_idx, true);
    auto* attn_res = OpFactory::add(attn_out, input, "attn_res", layer_idx);

    auto* ffn_norm_w = OpFactory::find_tensor(info, p + ".post_attention_norm.weight");
    auto* ffn_normed = OpFactory::rms_norm(attn_res, ffn_norm_w, config_.rms_norm_eps, "ffn_normed", layer_idx);
    auto* ffn_gate_w = OpFactory::find_tensor(info, p + ".ffn_gate.weight");
    auto* ffn_up_w   = OpFactory::find_tensor(info, p + ".ffn_up.weight");
    auto* ffn_gate = OpFactory::linear(ffn_normed, ffn_gate_w, "ffn_gate", layer_idx, true);
    auto* ffn_up   = OpFactory::linear(ffn_normed, ffn_up_w,   "ffn_up",   layer_idx, true);
    auto* ffn_inter = OpFactory::mul(OpFactory::silu(ffn_gate, "ffn_gate_act", layer_idx), ffn_up, "ffn_inter", layer_idx);
    auto* ffn_down_w = OpFactory::find_tensor(info, p + ".ffn_down.weight");
    auto* ffn_out = OpFactory::linear(ffn_inter, ffn_down_w, "ffn_out", layer_idx, true);
    
    return OpFactory::add(ffn_out, attn_res, "layer_out", layer_idx);
}