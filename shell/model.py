import yaml
import torch
import torch.nn.functional as F
from typing import List, Optional

from ops import (
    embedding_lookup,
    rms_norm,
    linear,
    silu,
    compute_rope_cos_sin,
    apply_rope,
    scaled_dot_product_attention,
)
from gguf import GGUFTorchLoader


class Qwen3Config:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.hidden_size: int = cfg["hidden_size"]
        self.num_layers: int = cfg["num_layers"]
        self.num_heads: int = cfg["num_heads"]
        self.num_kv_heads: int = cfg["num_kv_heads"]
        self.head_dim: int = cfg["head_dim"]
        self.intermediate_size: int = cfg["intermediate_size"]
        self.vocab_size: int = cfg["vocab_size"]
        self.max_seq_len: int = cfg["max_seq_len"]
        self.rms_norm_eps: float = cfg["rms_norm_eps"]
        self.rope_theta: float = cfg["rope_theta"]


class Qwen3Attention:
    def __init__(self, config: Qwen3Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = f"blk.{layer_idx}"
        # lazy-loaded weights
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.o_weight = None
        self.q_norm_weight = None
        self.k_norm_weight = None
        self.attn_norm_weight = None

    def load_weights(self, loader: GGUFTorchLoader):
        p = self.prefix
        self.q_weight = loader.load_tensor(f"{p}.attn_q.weight")
        self.k_weight = loader.load_tensor(f"{p}.attn_k.weight")
        self.v_weight = loader.load_tensor(f"{p}.attn_v.weight")
        self.o_weight = loader.load_tensor(f"{p}.attn_output.weight")
        self.q_norm_weight = loader.load_tensor(f"{p}.attn_q_norm.weight")
        self.k_norm_weight = loader.load_tensor(f"{p}.attn_k_norm.weight")
        self.attn_norm_weight = loader.load_tensor(f"{p}.attn_norm.weight")

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config

        # 1. RMSNorm
        x_norm = rms_norm(x, self.attn_norm_weight, eps=cfg.rms_norm_eps)
        print(x_norm)

        # 2. Q/K/V projections
        q_flat = linear(x_norm, self.q_weight, None)  # [S, num_heads*head_dim]
        k_flat = linear(x_norm, self.k_weight, None)  # [S, num_kv_heads*head_dim]
        v_flat = linear(x_norm, self.v_weight, None)  # [S, num_kv_heads*head_dim]

        print(k_flat)

        # 3. Reshape to 4D: [1, S, hidden] -> [1, n_heads, S, head_dim]
        q_4d = q_flat.view(1, -1, cfg.num_heads, cfg.head_dim).permute(0, 2, 1, 3).contiguous()
        k_4d = k_flat.view(1, -1, cfg.num_kv_heads, cfg.head_dim).permute(0, 2, 1, 3).contiguous()
        v_4d = v_flat.view(1, -1, cfg.num_kv_heads, cfg.head_dim).permute(0, 2, 1, 3).contiguous()

        # 4. Per-head RMSNorm
        q_normed = rms_norm(q_4d, self.q_norm_weight, eps=cfg.rms_norm_eps)
        k_normed = rms_norm(k_4d, self.k_norm_weight, eps=cfg.rms_norm_eps)

        # 5. RoPE
        q_rope = apply_rope(q_normed, cos, sin, position_ids)
        k_rope = apply_rope(k_normed, cos, sin, position_ids)

        # 6. SDPA with causal mask (GQA)
        attn_out = scaled_dot_product_attention(q_rope, k_rope, v_4d, is_causal=True)

        # 7. Reshape back: [1, n_heads, S, head_dim] -> [S, n_heads*head_dim]
        attn_flat = attn_out.permute(0, 2, 1, 3).contiguous().view(1, -1, cfg.num_heads * cfg.head_dim)

        # 8. Output projection
        attn_out = linear(attn_flat, self.o_weight, None)  # [1, S, hidden_size]

        return attn_out


class Qwen3FFN:
    def __init__(self, config: Qwen3Config, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = f"blk.{layer_idx}"
        self.gate_weight = None
        self.up_weight = None
        self.down_weight = None
        self.ffn_norm_weight = None

    def load_weights(self, loader: GGUFTorchLoader):
        p = self.prefix
        self.gate_weight = loader.load_tensor(f"{p}.ffn_gate.weight")
        self.up_weight = loader.load_tensor(f"{p}.ffn_up.weight")
        self.down_weight = loader.load_tensor(f"{p}.ffn_down.weight")
        self.ffn_norm_weight = loader.load_tensor(f"{p}.ffn_norm.weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        # 1. RMSNorm
        x_norm = rms_norm(x, self.ffn_norm_weight, eps=cfg.rms_norm_eps)

        # 2. SwiGLU: gate + up
        gate = linear(x_norm, self.gate_weight, None)
        up = linear(x_norm, self.up_weight, None)

        # 3. SiLU(gate) * up
        ffn_inter = silu(gate) * up

        # 4. Down projection
        return linear(ffn_inter, self.down_weight, None)


class Qwen3Block:
    def __init__(self, config: Qwen3Config, layer_idx: int):
        self.config = config
        self.attn = Qwen3Attention(config, layer_idx)
        self.ffn = Qwen3FFN(config, layer_idx)

    def load_weights(self, loader: GGUFTorchLoader):
        self.attn.load_weights(loader)
        self.ffn.load_weights(loader)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention + residual
        attn_out = self.attn.forward(x, cos, sin, position_ids)
        x = x + attn_out
        # FFN + residual
        ffn_out = self.ffn.forward(x)
        x = x + ffn_out
        return x


class Qwen3Model:
    def __init__(self, config: Qwen3Config):
        self.config = config
        self.blocks = [Qwen3Block(config, i) for i in range(config.num_layers)]
        self.token_embd_weight = None
        self.output_norm_weight = None
        self.cos = None
        self.sin = None
    def load_weights(self, loader: GGUFTorchLoader):
        # Embedding: GGUF [hidden_size, vocab_size] -> transpose to [vocab_size, hidden_size]
        self.token_embd_weight = loader.load_tensor("token_embd.weight")
        self.output_norm_weight = loader.load_tensor("output_norm.weight")
        for block in self.blocks:
            block.load_weights(loader)
        # Precompute RoPE cache
        cfg = self.config
        self.cos, self.sin = compute_rope_cos_sin(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
    def forward(
        self,
        input_ids: List[int],
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config
        # 1. Embedding lookup
        x = embedding_lookup(self.token_embd_weight, input_ids)  # [S, hidden_size]
        x = x.unsqueeze(0)  # [1, S, hidden_size]
        # 2. Transformer blocks
        for block in self.blocks:
            x = block.forward(x, self.cos, self.sin, position_ids)
        # 3. Final RMSNorm
        x = rms_norm(x, self.output_norm_weight, eps=cfg.rms_norm_eps)
        # 4. LM Head (reuse embedding weight, transpose back for linear)
        logits = linear(x, self.token_embd_weight, None)  # [1, S, vocab_size]
        return logits
