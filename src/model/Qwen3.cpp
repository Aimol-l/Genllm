#include "model/Qwen3.hpp"
#include "memory_manager.h"
#include "model/op_factory.hpp"
#include <fstream>


void Qwen3Model::print_info(){
    std::println("=== Qwen3 Model Info ===");
    std::println("  Name: {}", name);
    std::println("  Type: {}", model_type_to_string(type));
    std::println("  Architecture: {}", model_arch_to_string(arch));
    std::println("  Padding: {}", params.padding);
}

void Qwen3Model::set_params(void* p){
    auto temp = static_cast<Qwen3Params*>(p);
    params.padding = temp->padding;
}
// ========== 辅助函数：构建单层 ==========
Tensor* Qwen3Model::build_qwen3_layer(
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
) {
    std::string prefix = std::format("blk.{}", layer_idx);
    // ──────────────────────────────────────────────────
    // [1] Self-Attention 分支
    // ──────────────────────────────────────────────────
    // 1.1 RMSNorm: input_layernorm
    const TensorInfo* attn_norm_info = OpFactory::find_tensor(info, prefix + ".attn_norm.weight");
    Tensor* x_norm = OpFactory::rms_norm(input, attn_norm_info, rms_norm_eps, "x_norm"); //[batch, seq_len, hidden_size]
    
    // 1.2 Q/K/V 投影 (并行)
    const TensorInfo* q_weight = OpFactory::find_tensor(info, prefix + ".attn_q.weight");
    const TensorInfo* k_weight = OpFactory::find_tensor(info, prefix + ".attn_k.weight");
    const TensorInfo* v_weight = OpFactory::find_tensor(info, prefix + ".attn_v.weight");

    Tensor* q_flat = OpFactory::linear(x_norm, q_weight, false, "q_flat"); //[B, S, 2048]
    Tensor* k_flat = OpFactory::linear(x_norm, k_weight, false, "k_flat");// k_flat: [B, S, 1024]
    Tensor* v_flat = OpFactory::linear(x_norm, v_weight, false, "v_flat");// v_flat: [B, S, 1024]
    
    // 1.3 Reshape + Permute: [B, S, n*head_dim] → [B, n, S, head_dim]
    Tensor* q_4d = OpFactory::reshape_permute(q_flat,{1,-1,num_heads,head_dim},{0, 2, 1, 3},"q_4d");
    Tensor* k_4d = OpFactory::reshape_permute(k_flat, {1, -1, num_kv_heads, head_dim}, {0, 2, 1, 3}, "k_4d");
    Tensor* v_4d = OpFactory::reshape_permute(v_flat, {1, -1, num_kv_heads, head_dim}, {0, 2, 1, 3}, "v_4d");
    
    // 1.4 Per-head RMSNorm (只对 head_dim 归一化)
    const TensorInfo* q_norm_info = OpFactory::find_tensor(info, prefix + ".attn_q_norm.weight");
    const TensorInfo* k_norm_info = OpFactory::find_tensor(info, prefix + ".attn_k_norm.weight");
    
    // [B, n, S, head_dim]
    Tensor* q_normed = OpFactory::rms_norm(q_4d, q_norm_info, rms_norm_eps, "q_normed");
    Tensor* k_normed = OpFactory::rms_norm(k_4d, k_norm_info, rms_norm_eps, "k_normed");

    // 1.5 Apply RoPE
    auto [q_rope, k_rope] = OpFactory::apply_rope(q_normed, k_normed, rope_cos, rope_sin);
    
    // 1.6 SDPA / FlashAttention
    Tensor* attn_4d = OpFactory::SDPA(q_rope, k_rope, v_4d,nullptr,-1.0f,true,16 / 8,"attn_out");
    
    // 1.7 Reshape back + Output projection, attn_flat: [B, S, 2048]
    Tensor* attn_flat = OpFactory::reshape_permute(attn_4d,{0, 2, 1, 3},{1, -1, num_heads * head_dim},"attn_flat");

    const TensorInfo* attn_out_weight = OpFactory::find_tensor(info, prefix + ".attn_output.weight");
    Tensor* attn_out = OpFactory::linear(attn_flat, attn_out_weight, false, "attn_out"); // attn_out: [B, S, hidden_size]
    
    // 1.8 残差连接
    Tensor* ffn_input = OpFactory::add(attn_out, input, "ffn_input"); //ffn_input: [B, S, hidden_size]
    // ──────────────────────────────────────────────────
    // [2] SwiGLU FFN 分支
    // ──────────────────────────────────────────────────
    // 2.1 RMSNorm: post_attention_layernorm
    const TensorInfo* ffn_norm_info = OpFactory::find_tensor(info, prefix + ".ffn_norm.weight");
    Tensor* ffn_normed = OpFactory::rms_norm(ffn_input, ffn_norm_info, rms_norm_eps, "ffn_normed");

    // 2.2 SwiGLU: gate + up (并行)
    const TensorInfo* gate_weight = OpFactory::find_tensor(info, prefix + ".ffn_gate.weight");
    const TensorInfo* up_weight = OpFactory::find_tensor(info, prefix + ".ffn_up.weight");
    Tensor* gate = OpFactory::linear(ffn_normed, gate_weight, false, "gate");// [B, S, 3072]
    Tensor* up = OpFactory::linear(ffn_normed, up_weight, false, "up"); // up: [B, S, 3072]
    
    // 2.3 SiLU(gate) * up
    Tensor* gate_act = OpFactory::silu(gate, "gate_act");
    Tensor* ffn_inter = OpFactory::mul(gate_act, up, "ffn_inter"); // ffn_inter: [B, S, 3072]

    // 2.4 Down projection + 残差
    const TensorInfo* down_weight = OpFactory::find_tensor(info, prefix + ".ffn_down.weight");
    Tensor* ffn_out = OpFactory::linear(ffn_inter, down_weight, false, "ffn_out"); // ffn_out: [B, S, hidden_size]
    Tensor* layer_output = OpFactory::add(ffn_out, ffn_input, "layer_output"); //[B, S, hidden_size]
    return layer_output;
}

ComputeGraph* Qwen3Model::build_graph(const GGUFInfo& info){
    std::println("Building Qwen3 computation graph...");
    // ========== Step 1: 解析配置参数 ==========
    auto& meta = info.metadata;
    // 从 GGUF metadata 读取配置
    int hidden_size = meta.value("qwen3.embedding_length", 1024);
    int num_layers = meta.value("qwen3.block_count", 28);
    int num_heads = meta.value("qwen3.attention.head_count", 16);
    int num_kv_heads = meta.value("qwen3.attention.head_count_kv", 16);
    int head_dim = hidden_size / num_heads;  // 128
    int vocab_size = meta.value("qwen3.vocab_size", 151936);
    int intermediate_size = meta.value("qwen3.feed_forward_length", 3072);
    float rms_norm_eps = meta.value("qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    float rope_theta = meta.value("qwen3.rope_freq_base", 1000000.0f);
    int max_seq_len = meta.value("qwen3.context_length", 40960);
    // ========== Step 2: 创建输入节点 ==========
    auto* graph = new ComputeGraph();
    // input_ids: [batch, seq_len] seq_len是token的数量
    Tensor* input_ids = OpFactory::placeholder(DataType::GGML_TYPE_I32,TensorType::TENSOR_TYPE_INPUT, {1, -1},"input_ids");
    graph->add_tensor(input_ids);  // 图接管所有权
    // ========== Step 3: 预计算 RoPE ==========
    // precompute_rope 只需要一次，与输入无关
    auto[rope_sin,rope_cos] = OpFactory::rope_cache(max_seq_len, head_dim, rope_theta,DataType::GGML_TYPE_F32);
    graph->add_tensor(rope_cos);
    graph->add_tensor(rope_sin);
    // ========== Step 4: 嵌入层 ==========
    const TensorInfo* embd_weight_info = OpFactory::find_tensor(info, "token_embd.weight");
    if (!embd_weight_info) {
        throw std::runtime_error("token_embd.weight not found in GGUF");
    }
    // Embedding lookup: input_ids → x_in [batch, seq_len, hidden_size]
    Tensor* x_in = OpFactory::embedding_lookup(input_ids, embd_weight_info,"x_in"); //{1, seq_len, hidden_size}
    graph->add_tensor(x_in);
    // ========== Step 5: 循环构建 28 层 ==========
    Tensor* prev_output = x_in;
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        prev_output = this->build_qwen3_layer(
            prev_output,
            info,
            layer_idx,
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_size,
            rms_norm_eps,
            rope_cos,
            rope_sin
        );
    }
    // ========== Step 6: 最终归一化 + LM Head ==========
    // 查找 output_norm.weight
    const TensorInfo* output_norm_info = OpFactory::find_tensor(info, "output_norm.weight");
    Tensor* final_norm = OpFactory::rms_norm(prev_output,output_norm_info,rms_norm_eps,"final_norm");
    // LM Head: Linear(final_norm, token_embd.weight.T)
    Tensor* logits = OpFactory::linear(final_norm,embd_weight_info,true,"logits"); //{1, seq_len, vocab_size}
    
    graph->add_tensor(final_norm);
    graph->add_tensor(logits);
    
    // ========== Step 8: 标记输入/输出 ==========
    graph->set_input("input_ids", input_ids);
    graph->set_output("logits", logits);
    
    // ========== Step 9: 构建执行顺序 (拓扑排序) ==========
    graph->build_exec_order();
    
    std::println("  ✓ Graph built: {} nodes", graph->tensor_count());
    return graph;  // 裸指针，所有权转移给调用者
}

void Qwen3Model::load_weights(GGUFInfo& info, MemoryManager* mem_manager) {
    if (!mem_manager) {
        throw std::runtime_error("Memory manager is null");
    }
    std::println("\n=== Loading Qwen3 weights ===");
    // TODO: 实现实际的权重加载逻辑
    // 1. 遍历 info.tensors_info，找到每个权重张量
    // 2. 从 graph.m_leafs 中找到对应的 Tensor 对象
    // 3. 打开 GGUF 文件
    // 4. 读取张量数据到已分配的内存（tensor->data）
    std::println("  Total tensors in GGUF: {}", info.tensors_info.size());
    std::println("  Total leaf tensors in graph: {}", graph.m_leafs.size());
    // 示例：遍历图中的叶子节点（权重）
    for (auto* leaf : graph.m_leafs) {
        if (!leaf || leaf->name.empty()) continue;
        std::println("  Found weight: {}", leaf->name);
        // 检查是否已分配内存
        if (leaf->data) {
            std::println("    Already allocated at 0x{:x}", reinterpret_cast<uintptr_t>(leaf->data));
        } else {
            std::println("    Not allocated yet");
        }
        // TODO: 从 GGUF 文件加载数据
        // 1. 在 info.tensors_info 中查找 leaf->name
        // 2. 获取张量的文件偏移量和大小
        // 3. 打开 GGUF 文件并读取数据到 leaf->data
    }
    std::println("\nWeight loading completed!");
}
