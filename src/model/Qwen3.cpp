#include "model/Qwen3.hpp"
#include "model/op_factory.hpp"


void Qwen3Model::print_info(){
    std::println("=== Qwen3 Model Info ===");
    std::println("  Name: {}", name);
    std::println("  Type: {}", model_type_to_string(type));
    std::println("  Architecture: {}", model_arch_to_string(arch));
    std::println("  hidden_size:  {}", config_.hidden_size);
    std::println("  num_layers:   {}", config_.num_layers);
    std::println("  num_heads:    {}", config_.num_heads);
    std::println("  num_kv_heads: {}", config_.num_kv_heads);
    std::println("  head_dim:     {}", config_.head_dim);
    std::println("  vocab_size:   {}", config_.vocab_size);
    std::println("  max_seq_len:  {}", config_.max_seq_len);
    std::println("  top_p: {}", params.top_p);
    std::println("  temperature: {}", params.templature);
}

void Qwen3Model::set_params(void* p){
    auto temp = static_cast<Qwen3Params*>(p);
    params.top_p = temp->top_p;
    params.templature = temp->templature;
    std::println("Updated Qwen3 parameters: top_p={}, temperature={}", params.top_p, params.templature);
}

void Qwen3Model::parse_config(const GGUFInfo& info) {
    auto& meta = info.metadata;
    config_.hidden_size       = meta.value("qwen3.embedding_length", 1024);
    config_.num_layers        = meta.value("qwen3.block_count", 28);
    config_.num_heads         = meta.value("qwen3.attention.head_count", 16);
    config_.num_kv_heads      = meta.value("qwen3.attention.head_count_kv", 8);
    config_.head_dim          = meta.value("qwen3.attention.key_length", 128);
    config_.intermediate_size = meta.value("qwen3.feed_forward_length", 3072);
    config_.rms_norm_eps      = meta.value("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    config_.rope_theta        = meta.value("qwen3.rope.freq_base", 1000000.0f);
    config_.max_seq_len       = meta.value("qwen3.context_length", 40960);

    // vocab_size 优先从 tokenizer.tokens 获取
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
    if (config_.vocab_size == 0)
        throw std::runtime_error("Cannot determine vocab_size");

    std::println("Config: hidden={}, heads={}/kv={}, head_dim={}, vocab={}, layers={}",
        config_.hidden_size, config_.num_heads, config_.num_kv_heads,
        config_.head_dim, config_.vocab_size, config_.num_layers);
}

// ========== 构建单层 ==========
Tensor* Qwen3Model::build_qwen3_layer(
    Tensor* input,
    const GGUFInfo& info,
    int layer_idx,
    Tensor* rope_cos,
    Tensor* rope_sin 
) {
    std::string prefix = std::format("blk.{}", layer_idx);

    // ──────────────────────────────────────────────────
    // [1] Self-Attention 分支
    // ──────────────────────────────────────────────────
    // 1.1 RMSNorm: input_layernorm
    const TensorInfo* attn_norm_info = OpFactory::find_tensor(info, prefix + ".attn_norm.weight");
    Tensor* x_norm = OpFactory::rms_norm(input, attn_norm_info, this->config_.rms_norm_eps, "x_norm",layer_idx);

    // 1.2 Q/K/V 投影 (并行)
    const TensorInfo* q_weight = OpFactory::find_tensor(info, prefix + ".attn_q.weight");
    const TensorInfo* k_weight = OpFactory::find_tensor(info, prefix + ".attn_k.weight");
    const TensorInfo* v_weight = OpFactory::find_tensor(info, prefix + ".attn_v.weight");

    Tensor* q_flat = OpFactory::linear(x_norm, q_weight, false, "q_flat",layer_idx);
    Tensor* k_flat = OpFactory::linear(x_norm, k_weight, false, "k_flat",layer_idx);
    Tensor* v_flat = OpFactory::linear(x_norm, v_weight, false, "v_flat",layer_idx);

    // 1.3 reshape + permute -> [B, n_heads, seq_len, head_dim]
    Tensor* q_4d = OpFactory::reshape_permute(q_flat, {1, -1, this->config_.num_heads,       this->config_.head_dim}, {0, 2, 1, 3}, "q_4d",layer_idx);
    Tensor* k_4d = OpFactory::reshape_permute(k_flat, {1, -1, this->config_.num_kv_heads,   this->config_.head_dim}, {0, 2, 1, 3}, "k_4d",layer_idx);
    Tensor* v_4d = OpFactory::reshape_permute(v_flat, {1, -1, this->config_.num_kv_heads,   this->config_.head_dim}, {0, 2, 1, 3}, "v_4d",layer_idx);

    // 1.4 Per-head RMSNorm
    const TensorInfo* q_norm_info = OpFactory::find_tensor(info, prefix + ".attn_q_norm.weight");
    const TensorInfo* k_norm_info = OpFactory::find_tensor(info, prefix + ".attn_k_norm.weight");

    Tensor* q_normed = OpFactory::rms_norm(q_4d, q_norm_info, this->config_.rms_norm_eps, "q_normed",layer_idx);
    Tensor* k_normed = OpFactory::rms_norm(k_4d, k_norm_info, this->config_.rms_norm_eps, "k_normed",layer_idx);

    auto [q_rope, k_rope] = OpFactory::apply_rope(q_normed, k_normed, rope_cos, rope_sin);

    // 1.6 SDPA / FlashAttention
    Tensor* attn_4d = OpFactory::SDPA(q_rope, k_rope, v_4d,nullptr,1.0f/(std::sqrt(static_cast<float>(this->config_.head_dim))),true, this->config_.head_dim / 8,"attn_4d",layer_idx);

    // 1.7 [B, n_heads, seq_len, head_dim] -> [B, seq_len, n_heads*head_dim]
    Tensor* attn_flat = OpFactory::permute_reshape(attn_4d,{0, 2, 1, 3},{1,-1,this->config_.num_heads * this->config_.head_dim},"attn_flat",layer_idx);

    const TensorInfo* attn_out_weight = OpFactory::find_tensor(info, prefix + ".attn_output.weight");
    Tensor* attn_out = OpFactory::linear(attn_flat, attn_out_weight, false, "attn_out",layer_idx);

    // 1.8 残差连接
    Tensor* ffn_input = OpFactory::add(attn_out, input, "ffn_input",layer_idx);
    // ──────────────────────────────────────────────────
    // [2] SwiGLU FFN 分支
    // ──────────────────────────────────────────────────
    const TensorInfo* ffn_norm_info = OpFactory::find_tensor(info, prefix + ".ffn_norm.weight");
    Tensor* ffn_normed = OpFactory::rms_norm(ffn_input, ffn_norm_info, this->config_.rms_norm_eps, "ffn_normed",layer_idx);

    const TensorInfo* gate_weight = OpFactory::find_tensor(info, prefix + ".ffn_gate.weight");
    const TensorInfo* up_weight   = OpFactory::find_tensor(info, prefix + ".ffn_up.weight");

    Tensor* gate = OpFactory::linear(ffn_normed, gate_weight, false, "gate",layer_idx);
    Tensor* up   = OpFactory::linear(ffn_normed, up_weight,   false, "up",layer_idx);

    Tensor* gate_act  = OpFactory::silu(gate, "gate_act",layer_idx);
    Tensor* ffn_inter = OpFactory::mul(gate_act, up, "ffn_inter",layer_idx);

    const TensorInfo* down_weight = OpFactory::find_tensor(info, prefix + ".ffn_down.weight");
    Tensor* ffn_out = OpFactory::linear(ffn_inter, down_weight, false, "ffn_out",layer_idx);
    Tensor* layer_output = OpFactory::add(ffn_out, ffn_input, "layer_output",layer_idx);

    return layer_output;
}

ComputeGraph& Qwen3Model::build_graph(const GGUFInfo& info){
    std::println("Building Qwen3 computation graph...");

    this->parse_config(info);
    this->config_.num_layers = 2; // 临时 hardcode 层数，方便测试。实际实现时应该使用 config_.num_layers

    Tensor* input_ids = OpFactory::placeholder(DataType::GGML_TYPE_I32,TensorType::TENSOR_TYPE_INPUT, {1, -1},"input_ids");

    auto[rope_sin, rope_cos] = OpFactory::rope_cache(this->config_.max_seq_len, this->config_.head_dim, this->config_.rope_theta, DataType::GGML_TYPE_F32);

    const TensorInfo* embd_weight_info = OpFactory::find_tensor(info, "token_embd.weight");

    if (!embd_weight_info) 
        throw std::runtime_error("token_embd.weight not found in GGUF");

    Tensor* x_in = OpFactory::embedding_lookup(input_ids, embd_weight_info, "x_in");

    Tensor* prev_output = x_in;
    for (int layer_idx = 0; layer_idx < this->config_.num_layers; ++layer_idx) {
        prev_output = this->build_qwen3_layer(
            prev_output, 
            info,
            layer_idx,
            rope_cos,
            rope_sin
        );
    }
    // ========== Step 6: 最终归一化 + LM Head ==========
    const TensorInfo* output_norm_info = OpFactory::find_tensor(info, "output_norm.weight");
    Tensor* final_norm = OpFactory::rms_norm(prev_output, output_norm_info, this->config_.rms_norm_eps, "final_norm", this->config_.num_layers);

    Tensor* logits = OpFactory::linear(final_norm, embd_weight_info, true, "logits", this->config_.num_layers);

    logits->type = TensorType::TENSOR_TYPE_OUTPUT;

    this->graph_.build_from_outputs({logits});
    return this->graph_;
}
