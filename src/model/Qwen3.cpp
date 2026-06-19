#include <memory>
#include <print>
#include "model/Qwen3.hpp"
#include "model/op_factory.hpp"
#include "utils/utils.hpp"


void Qwen3Model::print_info(){
    std::println("=== Qwen3 Model Info ===");
    std::println("  Name: {}", name);
    std::println("  Type: {}", model_type_to_string(type));
    std::println("  Architecture: {}", model_arch_to_string(arch));
    std::println("  embedding_length:  {}", config_.embedding_length);
    std::println("  block_count:   {}", config_.block_count);
    std::println("  head_count:    {}", config_.head_count);
    std::println("  num_kv_heads: {}", config_.num_kv_heads);
    std::println("  key_length:     {}", config_.key_length);
    std::println("  vocab_size:   {}", config_.vocab_size);
    std::println("  context_length:  {}", config_.context_length);
}

void Qwen3Model::parse_config(const GGUFInfo& info) {
    auto& meta = info.metadata;
    config_.block_count        = meta.value("qwen3.block_count", 28);
    config_.context_length     = meta.value("qwen3.context_length", 40960);
    config_.embedding_length   = meta.value("qwen3.embedding_length", -1);
    config_.feed_forward_length= meta.value("qwen3.feed_forward_length", 3072);
    config_.head_count         = meta.value("qwen3.attention.head_count", 16);
    config_.num_kv_heads       = meta.value("qwen3.attention.head_count_kv", 8);
    config_.key_length         = meta.value("qwen3.attention.key_length", 128);
    config_.intermediate_size  = meta.value("qwen3.feed_forward_length", 3072);
    config_.rms_norm_eps       = meta.value("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    config_.freq_base          = meta.value("qwen3.rope.freq_base", 1000000.0f);

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

    std::println("Config: hidden={}, heads={}/kv={}, key_length={}, vocab={}, layers={}",
        config_.embedding_length, config_.head_count, config_.num_kv_heads,
        config_.key_length, config_.vocab_size, config_.block_count);
}

std::unique_ptr<ComputeGraph> Qwen3Model::build_graph(const GGUFInfo& info){
    std::println("Building Qwen3 computation graph...");
    // ========== Step 1: 解析配置参数 ==========
    this->parse_config(info);
    // this->config_.block_count = 1; // 临时测试
    //=====================================================================================
    Tensor* input_ids = OpFactory::placeholder(DataType::GGML_TYPE_I32,TensorType::TENSOR_TYPE_INPUT, {1, -1},"input_ids",-1); //[B, seq_len]

    // [context_length, key_length]
    auto [rope_cos, rope_sin] = OpFactory::rope_cache(
        config_.context_length,
        config_.key_length,
        config_.freq_base,
        DataType::GGML_TYPE_F32,
        "rope",-1
    ); 

    const TensorInfo* embd_weight_info = OpFactory::find_tensor(info, "token_embd.weight"); // [vocab_size, embedding_length]
    const TensorInfo* output_norm_info = OpFactory::find_tensor(info, "output_norm.weight"); // [embedding_length]

    if (!embd_weight_info) throw std::runtime_error("token_embd.weight not found in GGUF");

    //[B, seq_len]，[vocab_size, embedding_length] -> [B, seq_len, embedding_length]
    // 对embd_weight_info进行转置标记，在算子实现的时候进行转置访问
    Tensor* x_in = OpFactory::embedding_lookup(input_ids, embd_weight_info,"x_in",-1,false);

    Tensor* prev_output = x_in; // [B, seq_len, embedding_length]

    for (int layer_idx = 0; layer_idx < config_.block_count; ++layer_idx) {
        prev_output = this->build_qwen3_layer(
            prev_output,
            info,
            layer_idx,
            config_.embedding_length,
            config_.head_count,
            config_.num_kv_heads,
            config_.key_length,
            config_.intermediate_size,
            config_.rms_norm_eps,
            rope_cos,
            rope_sin
        ); // [B, seq_len, embedding_length]
    }
    // ========== Step 6: 最终归一化 + LM Head ==========
    Tensor* final_norm = OpFactory::rms_norm(prev_output,output_norm_info,config_.rms_norm_eps,"final_norm",config_.block_count); // [B, seq_len, embedding_length]

    // LM Head: [B, seq_len, embedding_length] @ [vocab_size, embedding_length] -> [B, seq_len, vocab_size]
    Tensor* logits = OpFactory::linear(final_norm,embd_weight_info,"logits",config_.block_count);
    
    // ========== Step 3: 从 logits 反向收集，构建 ComputeGraph ==========
    auto graph = std::make_unique<ComputeGraph>();
    graph->build_from_outputs({logits});
    return graph;
}

Tensor* Qwen3Model::build_qwen3_layer(
    Tensor* input,           // [batch, seq_len, embedding_length]
    const GGUFInfo& info,
    int layer_idx,
    int embedding_length,
    int head_count,
    int num_kv_heads,
    int key_length,
    int intermediate_size,
    float rms_norm_eps,
    Tensor* rope_cos,
    Tensor* rope_sin 
) {
    std::string prefix = std::format("blk.{}", layer_idx);
    // ──────────────────────────────────────────────────
    // [1] Self-Attention 分支
    // ──────────────────────────────────────────────────
    // 1.1 RMSNorm: input_layernorm
    const TensorInfo* attn_norm_info = OpFactory::find_tensor(info, prefix + ".attn_norm.weight"); // [embedding_length]
    // [batch, seq_len, embedding_length],[embedding_length] -> [batch, seq_len, embedding_length ]
    Tensor* x_norm = OpFactory::rms_norm(input, attn_norm_info, rms_norm_eps, "x_norm",layer_idx); // [batch, seq_len, embedding_length]
    
    // 1.2 Q/K/V 投影
    const TensorInfo* q_weight = OpFactory::find_tensor(info, prefix + ".attn_q.weight"); // [2048, embedding_length]
    const TensorInfo* k_weight = OpFactory::find_tensor(info, prefix + ".attn_k.weight"); // [1024, embedding_length]
    const TensorInfo* v_weight = OpFactory::find_tensor(info, prefix + ".attn_v.weight"); // [1024, embedding_length]

    // [batch, seq_len, embedding_length] @ [2048, embedding_length] -> [batch, seq_len, 2048], 这种张量排列更有利于计算的时候缓存命中
    Tensor* q_flat = OpFactory::linear(x_norm, q_weight, "q_flat",layer_idx);
    // [batch, seq_len, embedding_length] @ [1024, embedding_length] -> [batch, seq_len, 1024]
    Tensor* k_flat = OpFactory::linear(x_norm, k_weight, "k_flat",layer_idx);
    // [batch, seq_len, embedding_length] @ [1024, embedding_length] -> [batch, seq_len, 1024]
    Tensor* v_flat = OpFactory::linear(x_norm, v_weight, "v_flat",layer_idx);
    
    // 1.3  
    // [batch, seq_len, 2048] -> [batch, seq_len, head_count, key_length] -> [batch, head_count, seq_len, key_length]
    Tensor* q_4d = OpFactory::reshape_permute(q_flat, {1, -1, head_count,    key_length}, {0, 2, 1, 3}, "q_4d",layer_idx);
    // [batch, seq_len, 1024] -> [batch, seq_len, num_kv_heads, key_length] -> [batch, num_kv_heads, seq_len, key_length]
    Tensor* k_4d = OpFactory::reshape_permute(k_flat, {1, -1, num_kv_heads, key_length}, {0, 2, 1, 3}, "k_4d",layer_idx);
    // [batch, seq_len, 1024] -> [batch, seq_len, num_kv_heads, key_length] -> [batch, num_kv_heads, seq_len, key_length]
    Tensor* v_4d = OpFactory::reshape_permute(v_flat, {1, -1, num_kv_heads, key_length}, {0, 2, 1, 3}, "v_4d",layer_idx);
    
    // 1.4 Per-head RMSNorm (只对 key_length 归一化)
    const TensorInfo* q_norm_info = OpFactory::find_tensor(info, prefix + ".attn_q_norm.weight"); // [128]
    const TensorInfo* k_norm_info = OpFactory::find_tensor(info, prefix + ".attn_k_norm.weight"); // [128]
    
    // [batch, head_count, seq_len, key_length] , [128] -> [batch, head_count, seq_len, key_length]
    Tensor* q_normed = OpFactory::rms_norm(q_4d, q_norm_info, rms_norm_eps, "q_normed",layer_idx); 
    // [batch, num_kv_heads, seq_len, key_length]
    Tensor* k_normed = OpFactory::rms_norm(k_4d, k_norm_info, rms_norm_eps, "k_normed",layer_idx); 

    // 1.5 Apply RoPE, [B, head_count, seq_len, key_length], [B, num_kv_heads, seq_len, key_length]
    auto [q_rope, k_rope] = OpFactory::apply_rope(q_normed, k_normed, rope_cos, rope_sin,nullptr,"",layer_idx); 
    
    // 1.6 Attention [B, head_count, seq_len, key_length]
    Tensor* attn_4d = OpFactory::Attention(
        q_rope, k_rope, v_4d,
        nullptr,
        1.0f/(std::sqrt(float(key_length))),
        true,
        config_.head_count / config_.num_kv_heads,
        "attn_4d",
        OperationType::OP_TYPE_PAGED_ATTN,
        layer_idx
    );
    
    // 1.7 [B, head_count, seq_len, key_length] -> [B, seq_len, head_count, key_length] -> [B, seq_len, head_count*key_length]
    Tensor* attn_flat = OpFactory::permute_reshape(attn_4d,{0, 2, 1, 3},{1,-1,head_count * key_length},"attn_flat",layer_idx);
    
    const TensorInfo* attn_out_weight = OpFactory::find_tensor(info, prefix + ".attn_output.weight");   // [embedding_length, 2048]

    // [B, seq_len, head_count * key_length]@ [embedding_length, 2048] -> [B, seq_len, embedding_length]
    Tensor* attn_out = OpFactory::linear(attn_flat, attn_out_weight, "attn_out",layer_idx);
    
    // 1.8 残差连接
    Tensor* ffn_input = OpFactory::add(attn_out, input, "ffn_input",layer_idx); // [B,seq_len, embedding_length]
    // ──────────────────────────────────────────────────
    // [2] SwiGLU FFN 分支
    // ──────────────────────────────────────────────────
    // 2.1 RMSNorm: post_attention_layernorm
    const TensorInfo* ffn_norm_info = OpFactory::find_tensor(info, prefix + ".ffn_norm.weight"); // [embedding_length]
    Tensor* ffn_normed = OpFactory::rms_norm(ffn_input, ffn_norm_info, rms_norm_eps, "ffn_normed",layer_idx); // [B, seq_len, embedding_length]

    // 2.2 SwiGLU: gate + up (并行)
    const TensorInfo* gate_weight = OpFactory::find_tensor(info, prefix + ".ffn_gate.weight");  // [3072, embedding_length]
    const TensorInfo* up_weight = OpFactory::find_tensor(info, prefix + ".ffn_up.weight");      // [3072, embedding_length]

    // [B, seq_len, embedding_length] @ [3072, embedding_length] -> [B, seq_len, 3072]
    Tensor* gate = OpFactory::linear(ffn_normed, gate_weight, "gate",layer_idx);
    // [B, seq_len, embedding_length] @ [3072, embedding_length] -> [B, seq_len, 3072]
    Tensor* up = OpFactory::linear(ffn_normed, up_weight, "up",layer_idx);
    
    // 2.3 SiLU(gate) * up
    Tensor* gate_act = OpFactory::silu(gate, "gate_act",layer_idx);// [B, seq_len, 3072]
    Tensor* ffn_inter = OpFactory::mul(gate_act, up, "ffn_inter",layer_idx); // [B, seq_len, 3072] * [B, seq_len, 3072] -> [B, seq_len, 3072]

    // 2.4 Down projection + 残差
    const TensorInfo* down_weight = OpFactory::find_tensor(info, prefix + ".ffn_down.weight");  // [embedding_length, 3072]

    // 2.5 [B, seq_len, 3072] @ [embedding_length, 3072] -> [B, seq_len, embedding_length]
    Tensor* ffn_out = OpFactory::linear(ffn_inter, down_weight, "ffn_out",layer_idx);
    Tensor* layer_output = OpFactory::add(ffn_out, ffn_input, "layer_output",layer_idx); // [B, seq_len, embedding_length]

    return layer_output;
}
