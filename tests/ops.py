import torch
import torch.nn.functional as F
from typing import Union, List
from typing import Optional



def embedding_lookup(weight: torch.Tensor, input_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
    """
    词嵌入查找 (等价于 torch.nn.Embedding)
    weight: (vocab_size, hidden_dim)  -> GGUF 中为 [1024, 151936]
    input_ids: (seq_len,)
    return: (seq_len, hidden_dim)
    """
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=weight.device)
    return F.embedding(input_ids, weight)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return norm_x * weight

def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor|None) -> torch.Tensor:
    return F.linear(x, weight, bias)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def compute_rope_cos_sin(seq_len: int, head_dim: int, rope_theta: float = 1000000.0, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算旋转位置编码的 cos 和 sin 缓存
    Args:
        seq_len: 序列长度（如 4096）
        head_dim: 注意力头维度（Qwen3-0.6B 通常为 64，即 1024/16）
        rope_theta: 基频率（Qwen3 默认 1e6）
    Returns:
        cos, sin: 形状均为 (seq_len, head_dim)，dtype=torch.float32
    """
    # 1. 计算逆频率: inv_freq[i] = 1 / (theta^(i/head_dim)), i ∈ {0, 2, 4, ..., head_dim-2}
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    
    # 2. 位置索引与逆频率的外积: (seq_len, head_dim//2)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    
    # 3. 拼接为完整维度并计算 cos/sin: (seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to Q or K.
    
    Args:
        x: [B, n_heads, seq_len, head_dim]
        cos: [max_seq_len, head_dim]
        sin: [max_seq_len, head_dim]
        position_ids: [seq_len] or [B, seq_len], optional
        
    Returns:
        x_rotated: [B, n_heads, seq_len, head_dim]
    """
    # ========== 1. 准备位置索引 ==========
    if position_ids is None:
        # 默认使用连续位置 [0, 1, ..., seq_len-1]
        seq_len = x.shape[2]
        position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
    
    # position_ids 形状标准化为 [seq_len]
    if position_ids.dim() == 2:
        position_ids = position_ids[0]  # 取 batch 第一行（假设 batch 内位置一致）
    
    # ========== 2. 切片获取当前序列的 cos/sin ==========
    # cos_cur: [seq_len, head_dim]
    cos_cur = cos[position_ids]
    sin_cur = sin[position_ids]

    # ========== 3. 维度广播对齐 ==========
    # x:        [B, n_heads, seq_len, head_dim]
    # cos_cur:  [1,       1,   seq_len, head_dim]
    cos_cur = cos_cur.unsqueeze(0).unsqueeze(0)
    sin_cur = sin_cur.unsqueeze(0).unsqueeze(0)

    # ========== 4. half-rotation (Qwen3/LLaMA 约定) ==========
    # 前半与后半配对: (x[i], x[i + d//2]) 用 freq_i 旋转
    half = x.shape[-1] // 2 #64
    x1 = x[..., :half]  #前半
    x2 = x[..., half:]  #后半
    rotated = torch.cat((-x2, x1), dim=-1)

    x_rotated = x * cos_cur + rotated * sin_cur
    
    return x_rotated


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 is_causal: bool = True) -> torch.Tensor:
    """
    缩放点积注意力 (SDPA)，支持 GQA（q_heads >= kv_heads）
    q: (batch, q_heads, seq_q, head_dim)
    k: (batch, kv_heads, seq_kv, head_dim)
    v: (batch, kv_heads, seq_kv, head_dim)
    is_causal: 是否使用因果掩码
    return: (batch, q_heads, seq_q, head_dim)
    """
    # GQA: 将 kv_heads 扩展到 q_heads（每个 kv 头被 n 个 q 头共享）
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    head_dim = q.shape[-1]
    scale = head_dim ** -0.5

    # (batch, n_heads, seq_q, seq_kv)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        seq_q = attn.shape[-2]
        seq_kv = attn.shape[-1]
        # 因果掩码：允许 pos_q <= pos_kv
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_kv, dtype=torch.bool, device=q.device),
            diagonal=seq_kv - seq_q + 1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))

    attn = F.softmax(attn, dim=-1)

    # (batch, n_heads, seq_q, seq_kv) @ (batch, n_heads, seq_kv, head_dim)
    out = torch.matmul(attn, v)
    return out