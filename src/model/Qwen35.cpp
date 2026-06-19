#include "model/Qwen35.hpp"
#include "utils/utils.hpp"
#include "model/op_factory.hpp"

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

std::unique_ptr<ComputeGraph> Qwen35Model::build_graph(const GGUFInfo& info){
    std::println("Building Qwen3.5 Hybrid (Mamba2+Attn) computation graph...");
    this->parse_config(info);
    // this->config_.block_count = 4; // 临时 hardcode 层数，方便测试。

    Tensor* input_ids = OpFactory::placeholder(DataType::GGML_TYPE_I32, TensorType::TENSOR_TYPE_INPUT, {1, -1}, "input_ids");

    auto [rope_cos, rope_sin] = OpFactory::rope_cache(
        config_.max_seq_len,
        config_.rope_dimension_count,
        config_.rope_theta,
        DataType::GGML_TYPE_F32
    );

    const TensorInfo* embd_weight_info = OpFactory::find_tensor(info, "token_embd.weight");

    if (!embd_weight_info)  
        throw std::runtime_error("token_embd.weight not found");

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
    Tensor* final_norm = OpFactory::rms_norm(prev_output, output_norm_info, config_.rms_norm_eps, "final_norm", config_.block_count,0.0f);

    Tensor* logits = OpFactory::linear(final_norm, embd_weight_info, "logits", config_.block_count);
    
    auto graph = std::make_unique<ComputeGraph>();
    graph->build_from_outputs({logits});
    return graph;
}

Tensor* Qwen35Model::build_ssm_layer(Tensor* input, const GGUFInfo& info, int layer_idx) {
    std::string p = std::format("blk.{}", layer_idx);

    auto* norm_w = OpFactory::find_tensor(info, p + ".attn_norm.weight");
    auto* x_norm = OpFactory::rms_norm(input, norm_w, config_.rms_norm_eps, "ssm_x_norm", layer_idx, 0.0f);

    auto* qkv_weight = OpFactory::find_tensor(info, p + ".attn_qkv.weight");
    auto* gate_weight = OpFactory::find_tensor(info, p + ".attn_gate.weight");

    auto* qkv = OpFactory::linear(x_norm, qkv_weight, "ssm_qkv", layer_idx);
    auto* z_gate   = OpFactory::linear(x_norm, gate_weight, "ssm_z_gate", layer_idx);

    auto* ssm_a_info     = OpFactory::find_tensor(info, p + ".ssm_a");
    auto* ssm_alpha = OpFactory::find_tensor(info, p + ".ssm_alpha.weight");
    auto* ssm_beta  = OpFactory::find_tensor(info, p + ".ssm_beta.weight");
    auto* ssm_dt    = OpFactory::find_tensor(info, p + ".ssm_dt.bias");

    auto* b_proj = OpFactory::linear(x_norm, ssm_beta,  "ssm_b_proj", layer_idx);
    auto* a_proj = OpFactory::linear(x_norm, ssm_alpha, "ssm_a_proj", layer_idx);

    auto* conv1d_weight = OpFactory::find_tensor(info, p + ".ssm_conv1d.weight");

    // [1,seq,conv_dim]
    auto* conv_out = OpFactory::causal_conv1d(
        qkv,
        conv1d_weight,
        config_.ssm_conv_kernel,
        "conv_out",
        layer_idx
    );

    // [1,seq,conv_dim]
    auto* conv_act = OpFactory::silu(conv_out, "conv_act", layer_idx);

    // 从权重形状反推维度，适配 0.8B (k=v) 和 4B (num_v=32 > num_k=16)
    // conv_dim = 2 * key_dim + value_dim，value_dim = ssm_inner_size
    // num_v_heads 从 ssm_a.shape[0] 来；num_k_heads = ssm_group_count
    int64_t conv_dim    = qkv_weight->dimensions[0];   // 6144 (0.8B) / 8192 (4B)
    int64_t value_dim   = config_.ssm_inner_size;      // 2048 / 4096
    int64_t key_dim     = (conv_dim - value_dim) / 2;  // 2048 / 2048
    int64_t num_k_heads = config_.ssm_group_count;     // 16 / 16
    int64_t num_v_heads = ssm_a_info->dimensions[0];   // 16 / 32
    int64_t k_head_dim  = key_dim / num_k_heads;       // 128 / 128
    int64_t v_head_dim  = value_dim / num_v_heads;     // 128 / 128
    // 当前 kernel 假设 k_head_dim == v_head_dim == D
    int64_t D = k_head_dim;

    auto* q = OpFactory::split(conv_act, 2, 0,           key_dim,   "ssm_Q_flat", layer_idx);
    auto* k = OpFactory::split(conv_act, 2, key_dim,     key_dim,   "ssm_K_flat", layer_idx);
    auto* v = OpFactory::split(conv_act, 2, 2 * key_dim, value_dim, "ssm_V_flat", layer_idx);

    auto* Q = OpFactory::reshape(q, {1, -1, num_k_heads, k_head_dim}, "ssm_Q", layer_idx);
    auto* K = OpFactory::reshape(k, {1, -1, num_k_heads, k_head_dim}, "ssm_K", layer_idx);
    auto* V = OpFactory::reshape(v, {1, -1, num_v_heads, v_head_dim}, "ssm_V", layer_idx);

    auto* ssm_raw = OpFactory::ssm_scan(
        Q, K, V,
        ssm_a_info,
        b_proj,
        a_proj,
        ssm_dt,
        num_k_heads, num_v_heads, D,
        "ssm_raw",
        layer_idx
    ); // [1, seq, num_v_heads, D]

    // RMSNormGated: rms_norm(ssm_output, norm_w) * silu(z_gate)
    auto* ssm_norm_w = OpFactory::find_tensor(info, p + ".ssm_norm.weight");

    auto* ssm_normed = OpFactory::rms_norm(ssm_raw, ssm_norm_w, config_.rms_norm_eps, "ssm_normed", layer_idx);
    auto* ssm_flat = OpFactory::reshape(ssm_normed, {1, -1, config_.ssm_inner_size}, "ssm_flat", layer_idx);
    
    auto* z_act = OpFactory::silu(z_gate, "z_act", layer_idx);
    auto* mixed = OpFactory::mul(ssm_flat, z_act, "ssm_mixed", layer_idx);

    auto* out_w = OpFactory::find_tensor(info, p + ".ssm_out.weight");
    auto* ssm_final = OpFactory::linear(mixed, out_w, "ssm_final", layer_idx);
    auto* attn_res = OpFactory::add(ssm_final, input, "ssm_res", layer_idx);


    auto* ffn_norm_weight = OpFactory::find_tensor(info, p + ".post_attention_norm.weight");
    auto* ffn_normed = OpFactory::rms_norm(attn_res, ffn_norm_weight, config_.rms_norm_eps, "ffn_normed", layer_idx,0.0f);

    auto* ffn_gate_w = OpFactory::find_tensor(info, p + ".ffn_gate.weight");
    auto* ffn_up_w   = OpFactory::find_tensor(info, p + ".ffn_up.weight");

    auto* ffn_gate = OpFactory::linear(ffn_normed, ffn_gate_w, "ffn_gate", layer_idx);
    auto* ffn_up   = OpFactory::linear(ffn_normed, ffn_up_w,   "ffn_up",   layer_idx);

    auto* ffn_inter = OpFactory::mul(OpFactory::silu(ffn_gate, "ffn_gate_act", layer_idx), ffn_up, "ffn_inter", layer_idx);

    auto* ffn_down_w = OpFactory::find_tensor(info, p + ".ffn_down.weight");
    auto* ffn_out = OpFactory::linear(ffn_inter, ffn_down_w, "ffn_out", layer_idx);
    
    return OpFactory::add(ffn_out, attn_res, "layer_out", layer_idx);
}

Tensor* Qwen35Model::build_attn_layer(Tensor* input, const GGUFInfo& info, int layer_idx, Tensor* rope_cos, Tensor* rope_sin) {
    std::string p = std::format("blk.{}", layer_idx);
    // attn_q.weight [4096, 1024] = [num_heads * 2 * head_dim, hidden]
    // GGUF stores HF-native interleaved: per-head packed [Q_h(256), gate_h(256)]
    int num_heads = config_.head_count;
    int num_kv_heads = config_.head_count_kv;
    int head_dim = config_.key_length;
    int q_dim = num_heads * head_dim;

    auto* norm_w = OpFactory::find_tensor(info, p + ".attn_norm.weight");
    auto* x_norm = OpFactory::rms_norm(input, norm_w, config_.rms_norm_eps, "attn_x_norm", layer_idx,0.0f);

    auto* q_w = OpFactory::find_tensor(info, p + ".attn_q.weight");
    auto* k_w = OpFactory::find_tensor(info, p + ".attn_k.weight");
    auto* v_w = OpFactory::find_tensor(info, p + ".attn_v.weight");
    // q_proj [B, S, num_heads * 2 * head_dim] — interleaved per-head
    auto* q_proj = OpFactory::linear(x_norm, q_w, "q_proj", layer_idx);

    // Reshape to [B, S, num_heads, 2*head_dim], then split last dim to extract Q and gate
    auto* q_reshaped = OpFactory::reshape(q_proj, {1, -1, num_heads, 2 * head_dim}, "q_reshaped", layer_idx);
    auto* q_h       = OpFactory::split(q_reshaped, 3, 0,        head_dim, "q_h",   layer_idx);  // [B, S, num_heads, head_dim]
    auto* gate_h    = OpFactory::split(q_reshaped, 3, head_dim, head_dim, "gate_h", layer_idx); // [B, S, num_heads, head_dim]

    // q_4d: [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
    auto* q_4d = OpFactory::permute(q_h, {0, 2, 1, 3}, "q_4d", layer_idx);
    // gate_flat: [B, S, num_heads, head_dim] -> [B, S, q_dim]
    auto* gate_flat = OpFactory::reshape(gate_h, {1, -1, q_dim}, "gate_flat", layer_idx);

    auto* k_flat = OpFactory::linear(x_norm, k_w, "k_flat", layer_idx);
    auto* v_flat = OpFactory::linear(x_norm, v_w, "v_flat", layer_idx);

    auto* k_4d = OpFactory::reshape_permute(k_flat, {1, -1, num_kv_heads, head_dim},          {0, 2, 1, 3}, "k_4d", layer_idx);
    auto* v_4d = OpFactory::reshape_permute(v_flat, {1, -1, num_kv_heads, config_.value_length}, {0, 2, 1, 3}, "v_4d", layer_idx);

    auto* q_norm_w = OpFactory::find_tensor(info, p + ".attn_q_norm.weight");
    auto* k_norm_w = OpFactory::find_tensor(info, p + ".attn_k_norm.weight");
    auto* q_normed = OpFactory::rms_norm(q_4d, q_norm_w, config_.rms_norm_eps, "q_normed", layer_idx, 0.0f);
    auto* k_normed = OpFactory::rms_norm(k_4d, k_norm_w, config_.rms_norm_eps, "k_normed", layer_idx, 0.0f);

    auto [q_rope, k_rope] = OpFactory::apply_rope(q_normed, k_normed, rope_cos, rope_sin, nullptr, "", layer_idx);

    float scale = 1.0f / std::sqrt(static_cast<float>(config_.key_length));

    auto* attn_4d = OpFactory::Attention(q_rope, k_rope, v_4d, nullptr, scale, true, config_.head_count / config_.head_count_kv, "attn_4d",OperationType::OP_TYPE_PAGED_ATTN, layer_idx);

    auto* attn_flat = OpFactory::permute_reshape(attn_4d, {0, 2, 1, 3}, {1, -1, q_dim}, "attn_flat", layer_idx);
    // gate: sigmoid(gate_flat) * attn_flat，再经过 o_proj
    auto* gate_act = OpFactory::sigmoid(gate_flat, "gate_act", layer_idx);
    auto* gated_attn = OpFactory::mul(gate_act, attn_flat, "gated_attn", layer_idx);
    auto* out_w = OpFactory::find_tensor(info, p + ".attn_output.weight");
    auto* attn_out = OpFactory::linear(gated_attn, out_w, "attn_out", layer_idx);
    auto* attn_res = OpFactory::add(attn_out, input, "attn_res", layer_idx);

    auto* ffn_norm_w = OpFactory::find_tensor(info, p + ".post_attention_norm.weight");
    auto* ffn_normed = OpFactory::rms_norm(attn_res, ffn_norm_w, config_.rms_norm_eps, "ffn_normed", layer_idx,0.0f);
    auto* ffn_gate_w = OpFactory::find_tensor(info, p + ".ffn_gate.weight");
    auto* ffn_up_w   = OpFactory::find_tensor(info, p + ".ffn_up.weight");
    auto* ffn_gate = OpFactory::linear(ffn_normed, ffn_gate_w, "ffn_gate", layer_idx);
    auto* ffn_up   = OpFactory::linear(ffn_normed, ffn_up_w,   "ffn_up",   layer_idx);
    auto* ffn_inter = OpFactory::mul(OpFactory::silu(ffn_gate, "ffn_gate_act", layer_idx), ffn_up, "ffn_inter", layer_idx);
    auto* ffn_down_w = OpFactory::find_tensor(info, p + ".ffn_down.weight");
    auto* ffn_out = OpFactory::linear(ffn_inter, ffn_down_w, "ffn_out", layer_idx);
    
    return OpFactory::add(ffn_out, attn_res, "layer_out", layer_idx);
}