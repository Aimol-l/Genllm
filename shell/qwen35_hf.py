import torch
torch.set_printoptions(precision=6, sci_mode=False, linewidth=200, threshold=1000)
from models.ops import compute_rope_cos_sin
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.Qwen35 import Qwen35Config, Qwen35Model, Qwen35SSMLayer, Qwen35AttentionLayer

MODEL_NAME = "weights/qwen35"


def _gemma_rmsnorm_weight(hf_weight: torch.Tensor) -> torch.Tensor:
    """Gemma3RMSNorm stores weight offset by 1: forward does (1 + w) * x. Our rms_norm does w * x, so we need to add 1."""
    return 1.0 + hf_weight.float()


def load_weights_from_hf(model: Qwen35Model, hf_model):
    config = model.config
    H = config.ssm_group_count   # 16
    D = config.ssm_state_size    # 128

    # Embedding: HF [vocab, hidden]
    model.token_embd_weight = hf_model.model.embed_tokens.weight.data.float().clone()
    print(f"\n\n[Embedding] token_embd_weight: shape={model.token_embd_weight.shape}, dtype={model.token_embd_weight.dtype}, "
          f"min={model.token_embd_weight.min().item():.6f}, max={model.token_embd_weight.max().item():.6f}, "
          f"mean={model.token_embd_weight.mean().item():.6f}")
    print(model.token_embd_weight)
    
    # Output norm (Gemma3RMSNorm)
    model.output_norm_weight = _gemma_rmsnorm_weight(hf_model.model.norm.weight.data) # type: ignore
    print(f"\n\n[OutputNorm] output_norm_weight: shape={model.output_norm_weight.shape}, dtype={model.output_norm_weight.dtype}, "
          f"min={model.output_norm_weight.min().item():.6f}, max={model.output_norm_weight.max().item():.6f}, "
          f"mean={model.output_norm_weight.mean().item():.6f}")
    print(model.output_norm_weight)
    
    for i in range(config.block_count):
        hl = hf_model.model.layers[i]
        layer = model.layers[i]

        # Norms (both are Gemma3RMSNorm with +1)
        attn_norm_w = _gemma_rmsnorm_weight(hl.input_layernorm.weight.data)
        ffn_norm_w = _gemma_rmsnorm_weight(hl.post_attention_layernorm.weight.data)

        print(f"\n\n--- Layer {i} ({'SSM' if isinstance(layer, Qwen35SSMLayer) else 'Attention'}) ---")

        if isinstance(layer, Qwen35SSMLayer):
            la = hl.linear_attn  # Qwen3_5GatedDeltaNet

            # HF Qwen3.5 has separate projections: in_proj_qkv (Q+K+V), in_proj_z, in_proj_b, in_proj_a
            # in_proj_qkv: [key_dim*2 + value_dim, hidden] = [6144, 1024]
            #   Layout: Q[0:2048], K[2048:4096], V[4096:6144]
            w_qkv = la.in_proj_qkv.weight.data.float()  # [6144, 1024]
            layer.qkv_weight = torch.nn.Parameter(w_qkv)
            print(f"\n  qkv_weight: shape={w_qkv.shape}, min={w_qkv.min().item():.6f}, max={w_qkv.max().item():.6f}, mean={w_qkv.mean().item():.6f}")
            print(w_qkv)

            # in_proj_z: [value_dim, hidden] = [2048, 1024]
            w_z = la.in_proj_z.weight.data.float()  # [2048, 1024]
            layer.gate_weight = torch.nn.Parameter(w_z)
            print(f"\n  gate_weight (z): shape={w_z.shape}, min={w_z.min().item():.6f}, max={w_z.max().item():.6f}, mean={w_z.mean().item():.6f}")
            print(w_z)

            # Norms
            layer.attn_norm_weight = torch.nn.Parameter(attn_norm_w)
            print(f"\n  attn_norm_weight: shape={attn_norm_w.shape}, min={attn_norm_w.min().item():.6f}, max={attn_norm_w.max().item():.6f}, mean={attn_norm_w.mean().item():.6f}")
            print(attn_norm_w)


            # Conv1d: [conv_dim, 1, kernel] → [conv_dim, kernel]
            w_conv = la.conv1d.weight.data.float().squeeze(1)
            layer.conv1d_weight = torch.nn.Parameter(w_conv)
            print(f"\n  conv1d_weight: shape={w_conv.shape}, min={w_conv.min().item():.6f}, max={w_conv.max().item():.6f}, mean={w_conv.mean().item():.6f}")
            print(w_conv)

            # A_log, dt_bias
            w_a_log = la.A_log.data.float().clone()
            layer.ssm_a = torch.nn.Parameter(-torch.exp(w_a_log))
            print(f"\n  ssm_a: shape={layer.ssm_a.shape}, min={layer.ssm_a.min().item():.6f}, max={layer.ssm_a.max().item():.6f}, mean={layer.ssm_a.mean().item():.6f}")
            print(layer.ssm_a)

            # in_proj_a: [num_v_heads, hidden] = [16, 1024]
            w_a = la.in_proj_a.weight.data.float()  # [16, 1024]
            layer.ssm_alpha_weight = torch.nn.Parameter(w_a)
            print(f"\n  ssm_alpha_weight (a): shape={w_a.shape}, min={w_a.min().item():.6f}, max={w_a.max().item():.6f}, mean={w_a.mean().item():.6f}")
            print(w_a)

            # in_proj_b: [num_v_heads, hidden] = [16, 1024]
            w_b = la.in_proj_b.weight.data.float()  # [16, 1024]
            layer.ssm_beta_weight = torch.nn.Parameter(w_b)
            print(f"\n  ssm_beta_weight (b): shape={w_b.shape}, min={w_b.min().item():.6f}, max={w_b.max().item():.6f}, mean={w_b.mean().item():.6f}")
            print(w_b)

            w_dt = la.dt_bias.data.float().clone()
            layer.ssm_dt_bias = torch.nn.Parameter(w_dt)
            print(f"\n  ssm_dt_bias: shape={w_dt.shape}, min={w_dt.min().item():.6f}, max={w_dt.max().item():.6f}, mean={w_dt.mean().item():.6f}")
            print(w_dt)

            # RMSNormGated weight (init=ones, forward=w*x, no +1)
            w_ssm_norm = la.norm.weight.data.float().clone()
            layer.ssm_norm_weight = torch.nn.Parameter(w_ssm_norm)
            print(f"\n  ssm_norm_weight: shape={w_ssm_norm.shape}, min={w_ssm_norm.min().item():.6f}, max={w_ssm_norm.max().item():.6f}, mean={w_ssm_norm.mean().item():.6f}")
            print(w_ssm_norm)

            # Output projection [hidden, value_dim]
            w_out = la.out_proj.weight.data.float().clone()
            layer.ssm_out_weight = torch.nn.Parameter(w_out)
            print(f"\n  ssm_out_weight: shape={w_out.shape}, min={w_out.min().item():.6f}, max={w_out.max().item():.6f}, mean={w_out.mean().item():.6f}")
            print(w_out)

            # FFN
            layer.ffn_norm_weight = torch.nn.Parameter(ffn_norm_w)
            print(f"\n  ffn_norm_weight: shape={ffn_norm_w.shape}, min={ffn_norm_w.min().item():.6f}, max={ffn_norm_w.max().item():.6f}, mean={ffn_norm_w.mean().item():.6f}")
            print(ffn_norm_w)

            w_gate = hl.mlp.gate_proj.weight.data.float().clone()
            layer.ffn_gate_weight = torch.nn.Parameter(w_gate)
            print(f"\n  ffn_gate_weight: shape={w_gate.shape}, min={w_gate.min().item():.6f}, max={w_gate.max().item():.6f}, mean={w_gate.mean().item():.6f}")
            print(w_gate)

            w_up = hl.mlp.up_proj.weight.data.float().clone()
            layer.ffn_up_weight = torch.nn.Parameter(w_up)
            print(f"\n  ffn_up_weight: shape={w_up.shape}, min={w_up.min().item():.6f}, max={w_up.max().item():.6f}, mean={w_up.mean().item():.6f}")
            print(w_up)

            w_down = hl.mlp.down_proj.weight.data.float().clone()
            layer.ffn_down_weight = torch.nn.Parameter(w_down)
            print(f"\n  ffn_down_weight: shape={w_down.shape}, min={w_down.min().item():.6f}, max={w_down.max().item():.6f}, mean={w_down.mean().item():.6f}")
            print(w_down)

        elif isinstance(layer, Qwen35AttentionLayer):
            sa = hl.self_attn  # Qwen3_5Attention

            # HF Q proj stored interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...]
            # forward() consumes this layout directly — no reorder needed
            w_q = sa.q_proj.weight.data.float()  # [4096, 1024]

            layer.attn_norm_weight = torch.nn.Parameter(attn_norm_w)
            print(f"\n  attn_norm_weight: shape={attn_norm_w.shape}, min={attn_norm_w.min().item():.6f}, max={attn_norm_w.max().item():.6f}, mean={attn_norm_w.mean().item():.6f}")
            print(attn_norm_w)

            layer.q_weight = torch.nn.Parameter(w_q)
            print(f"\n  q_weight: shape={w_q.shape}, min={w_q.min().item():.6f}, max={w_q.max().item():.6f}, mean={w_q.mean().item():.6f}")
            print(w_q)

            w_k = sa.k_proj.weight.data.float().clone()
            layer.k_weight = torch.nn.Parameter(w_k)
            print(f"\n  k_weight: shape={w_k.shape}, min={w_k.min().item():.6f}, max={w_k.max().item():.6f}, mean={w_k.mean().item():.6f}")
            print(w_k)

            w_v = sa.v_proj.weight.data.float().clone()
            layer.v_weight = torch.nn.Parameter(w_v)
            print(f"\n  v_weight: shape={w_v.shape}, min={w_v.min().item():.6f}, max={w_v.max().item():.6f}, mean={w_v.mean().item():.6f}")
            print(w_v)

            w_o = sa.o_proj.weight.data.float().clone()
            layer.o_weight = torch.nn.Parameter(w_o)
            print(f"\n  o_weight: shape={w_o.shape}, min={w_o.min().item():.6f}, max={w_o.max().item():.6f}, mean={w_o.mean().item():.6f}")
            print(w_o)

            # QK norm (Gemma3RMSNorm with +1)
            w_q_norm = _gemma_rmsnorm_weight(sa.q_norm.weight.data)
            layer.q_norm_weight = torch.nn.Parameter(w_q_norm)
            print(f"\n  q_norm_weight: shape={w_q_norm.shape}, min={w_q_norm.min().item():.6f}, max={w_q_norm.max().item():.6f}, mean={w_q_norm.mean().item():.6f}")
            print(w_q_norm)

            w_k_norm = _gemma_rmsnorm_weight(sa.k_norm.weight.data)
            layer.k_norm_weight = torch.nn.Parameter(w_k_norm)
            print(f"\n  k_norm_weight: shape={w_k_norm.shape}, min={w_k_norm.min().item():.6f}, max={w_k_norm.max().item():.6f}, mean={w_k_norm.mean().item():.6f}")
            print(w_k_norm)

            # Norms
            layer.ffn_norm_weight = torch.nn.Parameter(ffn_norm_w)
            print(f"\n  ffn_norm_weight: shape={ffn_norm_w.shape}, min={ffn_norm_w.min().item():.6f}, max={ffn_norm_w.max().item():.6f}, mean={ffn_norm_w.mean().item():.6f}")
            print(ffn_norm_w)

            # FFN
            w_gate = hl.mlp.gate_proj.weight.data.float().clone()
            layer.ffn_gate_weight = torch.nn.Parameter(w_gate)
            print(f"\n  ffn_gate_weight: shape={w_gate.shape}, min={w_gate.min().item():.6f}, max={w_gate.max().item():.6f}, mean={w_gate.mean().item():.6f}")
            print(w_gate)

            w_up = hl.mlp.up_proj.weight.data.float().clone()
            layer.ffn_up_weight = torch.nn.Parameter(w_up)
            print(f"\n  ffn_up_weight: shape={w_up.shape}, min={w_up.min().item():.6f}, max={w_up.max().item():.6f}, mean={w_up.mean().item():.6f}")
            print(w_up)

            w_down = hl.mlp.down_proj.weight.data.float().clone()
            layer.ffn_down_weight = torch.nn.Parameter(w_down)
            print(f"\n  ffn_down_weight: shape={w_down.shape}, min={w_down.min().item():.6f}, max={w_down.max().item():.6f}, mean={w_down.mean().item():.6f}")
            print(w_down)

    # RoPE cache
    model.cos, model.sin = compute_rope_cos_sin(
        config.max_seq_len, config.rope_dimension_count, config.rope_theta
    )


if __name__ == "__main__":
    # config 直接从 GGUF 元数据解析；权重仍从 HF safetensors 加载
    config = Qwen35Config("weights/qwen35/Qwen3.5-0.8B-BF16.gguf")
    print(f"\nConfig: hidden={config.embedding_length}, layers={config.block_count}, "
          f"heads={config.head_count}/{config.head_count_kv}, vocab={config.vocab_size}")

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True)
    hf_model.eval()

    # 3. Build model & load weights from HF
    model = Qwen35Model(config)
    load_weights_from_hf(model, hf_model)

    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt = "1+1=?"

    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        enable_thinking = False,
        add_generation_prompt=True
    )
    chat_ids = chat_ids["input_ids"] if hasattr(chat_ids, "input_ids") else chat_ids

    input_ids = list(chat_ids)

    EOS_TOKEN_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}

    model.reset_states()
    with torch.inference_mode():
        # Prefill
        logits = model.forward(input_ids)
        next_token = logits[0, -1].argmax().item()

        # Decode
        for step in range(127):
            if next_token in EOS_TOKEN_IDS:
                break
            print(tokenizer.decode([next_token]), end="", flush=True)
            input_ids.append(next_token)
            logits = model.forward([next_token])
            next_token = logits[0, -1].argmax().item()

    print("\n")
    print("=========================")