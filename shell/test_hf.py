import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import Qwen3Config, Qwen3Model
from ops import compute_rope_cos_sin

MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_weights_from_hf(model: Qwen3Model, hf_model):
    """从 HuggingFace 模型加载权重到 Qwen3Model"""
    config = model.config

    # Embedding: HF [vocab, hidden] -> our [vocab, hidden]
    model.token_embd_weight = hf_model.model.embed_tokens.weight.data.clone()

    print(model.token_embd_weight.shape)
    print(model.token_embd_weight)

    # Output norm
    model.output_norm_weight = hf_model.model.norm.weight.data.clone()

    for i in range(config.num_layers):
        hl = hf_model.model.layers[i]
        b = model.blocks[i]

        # Attention - HF 存 [out, in], our linear 做 x @ weight, 需要 [in, out]
        b.attn.q_weight = hl.self_attn.q_proj.weight.data.T.contiguous()
        b.attn.k_weight = hl.self_attn.k_proj.weight.data.T.contiguous()
        b.attn.v_weight = hl.self_attn.v_proj.weight.data.T.contiguous()
        b.attn.o_weight = hl.self_attn.o_proj.weight.data.T.contiguous()
        b.attn.attn_norm_weight = hl.input_layernorm.weight.data.clone()
        b.attn.q_norm_weight = hl.self_attn.q_norm.weight.data.clone()
        b.attn.k_norm_weight = hl.self_attn.k_norm.weight.data.clone()

        # FFN
        b.ffn.gate_weight = hl.mlp.gate_proj.weight.data.T.contiguous()
        b.ffn.up_weight = hl.mlp.up_proj.weight.data.T.contiguous()
        b.ffn.down_weight = hl.mlp.down_proj.weight.data.T.contiguous()
        b.ffn.ffn_norm_weight = hl.post_attention_layernorm.weight.data.clone()

        print(b.ffn.down_weight.shape)
        print(b.ffn.down_weight)

    # RoPE cache
    model.cos, model.sin = compute_rope_cos_sin(
        config.max_seq_len, config.head_dim, config.rope_theta
    )


if __name__ == "__main__":
    # 1. 加载配置
    config = Qwen3Config("models/qwen3.yaml")
    print(f"Config: hidden={config.hidden_size}, layers={config.num_layers}, "
          f"heads={config.num_heads}/{config.num_kv_heads}, vocab={config.vocab_size}")

    # 2. 从 HuggingFace 加载权重
    print(f"Loading HF model: {MODEL_NAME} ...")
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    hf_model.eval()

    # 3. 构建模型 & 加载权重
    model = Qwen3Model(config)
    load_weights_from_hf(model, hf_model)

    # 4. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 5. 前向推理 (chat 格式)
    prompt = "1+1="
    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    chat_ids = chat_ids["input_ids"] if hasattr(chat_ids, "input_ids") else chat_ids

    print(f"Input:  {prompt!r}")
    print(f"Token ids: {chat_ids}")

    # 6. Greedy decode
    input_ids = list(chat_ids)
    max_new_tokens = 256
    EOS_TOKEN_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    print(f"Generating (max {max_new_tokens} tokens) ...")
    for step in range(max_new_tokens):
        logits = model.forward(input_ids)
        next_token = logits[0, -1].argmax().item()

        if next_token in EOS_TOKEN_IDS:
            break

        input_ids.append(next_token)

        # 实时打印当前 token
        print(tokenizer.decode([next_token]), end="", flush=True)

    print("\n")

    print("=========================")
    # 7. 完整输出
    generated = tokenizer.decode(input_ids)
    print(f"Full output:\n{generated}")

    # 8. 与 HF 模型对比验证 (仅最后一步)
    with torch.no_grad():
        hf_logits = hf_model(torch.tensor([input_ids])).logits
    diff = (hf_logits[0, -1] - logits[0, -1]).abs()
    print(f"\nHF logits diff (last token): max={diff.max():.6f}, mean={diff.mean():.6f}")
