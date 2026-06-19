import sys
import numpy as np
import torch
from typing import Optional


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
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
    # x:        [B, num_heads, seq_len, head_dim]
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



if __name__ == "__main__":
    x       = torch.from_numpy(np.load("dump/x.npy"))
    cos     = torch.from_numpy(np.load("dump/rope_cos.npy"))
    sin     = torch.from_numpy(np.load("dump/rope_sin.npy"))
    out_cuda = torch.from_numpy(np.load("dump/out.npy"))

    rope_dim = cos.shape[-1]
    head_dim = x.shape[-1]

    ref = torch.cat([
        apply_rope(x[..., :rope_dim], cos, sin),
        x[..., rope_dim:]
    ], dim=-1) if rope_dim < head_dim else apply_rope(x, cos, sin)

    print(ref)
    print(out_cuda)

    diff = (out_cuda - ref).abs()
    max_err = diff.max().item()
    print(f"MaxAbsErr: {max_err:.8f}")
    if max_err < 1e-5:
        print(">>> PASS")
    elif torch.allclose(out_cuda, ref, atol=1e-4, rtol=1e-4):
        print(">>> PASS (atol=1e-4)")
    else:
        print(">>> FAIL")
        sys.exit(1)
