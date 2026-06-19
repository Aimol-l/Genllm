import sys
import torch
import torch.nn.functional as F
import numpy as np

np.set_printoptions(suppress=True, precision=6, linewidth=200)

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


if __name__ == "__main__":
    Q = torch.from_numpy(np.load("dump/Q.npy"))
    K = torch.from_numpy(np.load("dump/K.npy"))
    V = torch.from_numpy(np.load("dump/V.npy"))
    out_cuda = torch.from_numpy(np.load("dump/Attn_out.npy"))

    # prefill → causal, decode(seq_q=1) → non-causal(已累加KV)
    is_causal = Q.shape[2] > 1
    ref = scaled_dot_product_attention(Q, K, V, is_causal)

    print(ref)
    print(out_cuda)

    diff = (out_cuda - ref).abs()
    max_err = diff.max().item()
    print(f"Q{K.shape} K{K.shape} V{V.shape} causal={is_causal}")
    print(f"MaxAbsErr: {max_err:.8f}")
    if max_err < 1e-5:
        print(">>> PASS")
    elif torch.allclose(out_cuda, ref, atol=1e-4, rtol=1e-4):
        print(">>> PASS (atol=1e-4)")
    else:
        print(">>> FAIL")
        sys.exit(1)
