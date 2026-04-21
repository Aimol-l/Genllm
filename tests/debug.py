import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from main import TENSOR_LIST
from model import Qwen3Config, Qwen3Model
from gguf import GGUFTorchLoader
from ops import *

DIR = os.path.dirname(os.path.abspath(__file__))
config = Qwen3Config(os.path.join(DIR, "../models/qwen3.yaml"))
loader = GGUFTorchLoader(os.path.join(DIR, "../models/Qwen3-0.6B-BF16.gguf"), 5951936, TENSOR_LIST)
model = Qwen3Model(config)
model.load_weights(loader)

p = [16, 10, 16, 28]

# Manual block 0
e = loader.load_tensor("token_embd.weight").T.contiguous()
x = embedding_lookup(e, p).unsqueeze(0)
xn = rms_norm(x, loader.load_tensor("blk.0.attn_norm.weight"), eps=1e-6)
q = linear(xn, loader.load_tensor("blk.0.attn_q.weight"), None)
k = linear(xn, loader.load_tensor("blk.0.attn_k.weight"), None)
v = linear(xn, loader.load_tensor("blk.0.attn_v.weight"), None)
q = q.view(1,-1,16,128).permute(0,2,1,3).contiguous()
k = k.view(1,-1,8,128).permute(0,2,1,3).contiguous()
v = v.view(1,-1,8,128).permute(0,2,1,3).contiguous()
q = rms_norm(q, loader.load_tensor("blk.0.attn_q_norm.weight"), eps=1e-6)
k = rms_norm(k, loader.load_tensor("blk.0.attn_k_norm.weight"), eps=1e-6)
cos, sin = compute_rope_cos_sin(40960, 128, 1000000)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
a = scaled_dot_product_attention(q, k, v, is_causal=True)
af = a.permute(0,2,1,3).contiguous().view(1,-1,2048)
ao = linear(af, loader.load_tensor("blk.0.attn_output.weight"), None)
r = x + ao
fn = rms_norm(r, loader.load_tensor("blk.0.ffn_norm.weight"), eps=1e-6)
g = linear(fn, loader.load_tensor("blk.0.ffn_gate.weight"), None)
u = linear(fn, loader.load_tensor("blk.0.ffn_up.weight"), None)
fo = linear(silu(g)*u, loader.load_tensor("blk.0.ffn_down.weight"), None)
manual = r + fo

# Model class block 0
mb0 = model.blocks[0].forward(x, model.cos, model.sin)
diff = (manual - mb0).abs()
print(f"Block0 manual: {manual.norm():.6f}, model: {mb0.norm():.6f}, max_diff: {diff.max():.2e}")
print(f"Match: {diff.max() < 1e-5}")

# Full model layer norms
x2 = embedding_lookup(model.token_embd_weight, p).unsqueeze(0)
for i, b in enumerate(model.blocks):
    x2 = b.forward(x2, model.cos, model.sin)
    print(f"L{i:2d}: {x2.norm():.4f} {'NaN!' if torch.isnan(x2).any() else ''}")
