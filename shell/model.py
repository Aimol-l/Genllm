import yaml
import torch
import torch.nn as nn
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


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = f"blk.{layer_idx}"

    def load_weights(self, loader: GGUFTorchLoader):
        p = self.prefix
        self.q_weight = nn.Parameter(loader.load_tensor(f"{p}.attn_q.weight"))
        self.k_weight = nn.Parameter(loader.load_tensor(f"{p}.attn_k.weight"))
        self.v_weight = nn.Parameter(loader.load_tensor(f"{p}.attn_v.weight"))
        self.o_weight = nn.Parameter(loader.load_tensor(f"{p}.attn_output.weight"))
        self.q_norm_weight = nn.Parameter(loader.load_tensor(f"{p}.attn_q_norm.weight"))
        self.k_norm_weight = nn.Parameter(loader.load_tensor(f"{p}.attn_k_norm.weight"))
        self.attn_norm_weight = nn.Parameter(loader.load_tensor(f"{p}.attn_norm.weight"))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config

        x_norm = rms_norm(x, self.attn_norm_weight, eps=cfg.rms_norm_eps)

        q_flat = linear(x_norm, self.q_weight, None)
        k_flat = linear(x_norm, self.k_weight, None)
        v_flat = linear(x_norm, self.v_weight, None)

        q_4d = q_flat.view(1, -1, cfg.num_heads, cfg.head_dim).permute(0, 2, 1, 3).contiguous()
        k_4d = k_flat.view(1, -1, cfg.num_kv_heads, cfg.head_dim).permute(0, 2, 1, 3).contiguous()
        v_4d = v_flat.view(1, -1, cfg.num_kv_heads, cfg.head_dim).permute(0, 2, 1, 3).contiguous()

        q_normed = rms_norm(q_4d, self.q_norm_weight, eps=cfg.rms_norm_eps)
        k_normed = rms_norm(k_4d, self.k_norm_weight, eps=cfg.rms_norm_eps)

        q_rope = apply_rope(q_normed, cos, sin, position_ids)
        k_rope = apply_rope(k_normed, cos, sin, position_ids)

        attn_out = scaled_dot_product_attention(q_rope, k_rope, v_4d, is_causal=True)

        attn_flat = attn_out.permute(0, 2, 1, 3).contiguous().view(1, -1, cfg.num_heads * cfg.head_dim)

        attn_out = linear(attn_flat, self.o_weight, None)

        return attn_out


class Qwen3FFN(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.prefix = f"blk.{layer_idx}"

    def load_weights(self, loader: GGUFTorchLoader):
        p = self.prefix
        self.gate_weight = nn.Parameter(loader.load_tensor(f"{p}.ffn_gate.weight"))
        self.up_weight = nn.Parameter(loader.load_tensor(f"{p}.ffn_up.weight"))
        self.down_weight = nn.Parameter(loader.load_tensor(f"{p}.ffn_down.weight"))
        self.ffn_norm_weight = nn.Parameter(loader.load_tensor(f"{p}.ffn_norm.weight"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config

        x_norm = rms_norm(x, self.ffn_norm_weight, eps=cfg.rms_norm_eps)

        gate = linear(x_norm, self.gate_weight, None)
        up = linear(x_norm, self.up_weight, None)

        ffn_inter = silu(gate) * up

        return linear(ffn_inter, self.down_weight, None)


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
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
        x = x + self.attn(x, cos, sin, position_ids)
        x = x + self.ffn(x)
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Qwen3Block(config, i) for i in range(config.num_layers)])

    def load_weights(self, loader: GGUFTorchLoader):
        self.token_embd_weight = nn.Parameter(loader.load_tensor("token_embd.weight"))
        self.output_norm_weight = nn.Parameter(loader.load_tensor("output_norm.weight"))
        for block in self.blocks:
            block.load_weights(loader)
        cfg = self.config
        cos, sin = compute_rope_cos_sin(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(
        self,
        input_ids: List[int],
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.config
        x = embedding_lookup(self.token_embd_weight, input_ids)
        x = x.unsqueeze(0)
        for block in self.blocks:
            x = block(x, self.cos, self.sin, position_ids)
        x = rms_norm(x, self.output_norm_weight, eps=cfg.rms_norm_eps)
        logits = linear(x, self.token_embd_weight, None)
        return logits
