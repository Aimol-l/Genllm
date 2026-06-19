#!/usr/bin/env python3
import numpy as np
import sys
import os

np.set_printoptions(suppress=True, precision=6, linewidth=200)

x = np.load("dump/linear_x.npy")
w = np.load("dump/linear_weight.npy")
ref = x @ w.T

print(ref)

# 如果有 C++ 输出文件，自动对比
cpp_path = "dump/linear_y_cpp.npy"
if os.path.exists(cpp_path):
    cpp = np.load(cpp_path)
    diff = np.abs(cpp.astype(np.float32) - ref)
    print()
    print(f"max abs err:  {diff.max():.10f}")
    print(f"mean abs err: {diff.mean():.10f}")
    print(f"cosine sim:   {np.dot(cpp.flatten(), ref.flatten()) / (np.linalg.norm(cpp) * np.linalg.norm(ref)):.10f}")
