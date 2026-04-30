# Genllm

基于 C++23 的模块化 LLM 推理框架，支持 CPU / CUDA 双后端，原生解析 GGUF 格式。

## 特性

- **GGUF 解析** — 完整的 GGUF v3 反序列化，支持 BPE tokenizer、BF16/F16/F32 数据类型
- **计算图 IR** — 基于 DAG 的可视化计算图，拓扑排序执行，支持 DOT 导出
- **分页注意力** — 分块 KV cache（PAGE_BLOCK_SIZE=16），支持多设备按层分配
- **双层内存池** — 权重/激活/KV cache 三池分离，激活池层间复用
- **自动调度** — 按显存容量自动分配连续层到 CPU/CUDA 设备，跨设备自动插入 copy 边
- **RoPE 缓存预计算** — 构建时预计算 cos/sin 表，推理时零拷贝复用
- **多模型兼容** — 模型工厂注册机制，支持 GQA（不同 head 比例的模型）

## 支持模型

| 模型 | 参数量 | 层数 | 隐藏维度 | 头数/KV头 | 推理状态 |
|------|--------|------|----------|-----------|---------|
| Qwen3-0.6B | 0.6B | 28 | 1024 | 16/8 | ✅ CPU + CUDA |
| Qwen3-1.7B | 1.7B | 28 | 2048 | 16/8 | ✅ CPU + CUDA |
| Qwen3-4B | 4B | 36 | 2560 | 32/8 | ✅ CPU + CUDA |

## 后端

| 后端 | 状态 | 已实现算子 |
|------|------|-----------|
| CPU | ✅ 完成 | linear, rms_norm, add/mul, silu, apply_rope, softmax, embedding, attention, reshape, permute, memcpy |
| CUDA | ✅ 完成 | 同上全部 + paged attention (cuBLAS GEMM) |
| SYCL | 🔜 计划 | — |
| Vulkan | 🔜 计划 | — |

## CUDA 算子实现

基于 warp-level 和 cuBLAS 实现，包括：
- **RMSNorm**: warp reduce (32 threads) 处理每行，支持 3D/4D 输入
- **Linear**: cublasGemmStridedBatchedEx + 缓存 handle 池
- **ApplyRoPE**: 每个线程处理 half_dim 对，cos/sin 表从 CPU 预计算后 D2H copy
- **Paged Attention**: 逐 block 遍历 page table，warp 处理单个 Q position，online softmax
- **Element-wise**: add/sub/mul/div/silu/gelu/relu 简单 grid-stride 并行

## 构建

```bash
# CPU only
cmake -B build -DBACKEND_CPU=ON -DBUILD_TEST=ON
cmake --build build -j

# CPU + CUDA (需要 CUDA Toolkit 13+)
cmake -B build -DBACKEND_CPU=ON -DBACKEND_CUDA=ON -DBUILD_TEST=ON
cmake --build build -j
```

## 运行

```bash
# 修改 tests/main.cpp 中的模型路径和 prompt
# 然后运行
./build/bin/main
```

关键配置（`tests/main.cpp`）：
- `kv_cache_per_layer`: 0=标准 attention, ≥1=分页 attention
- `max_seq_len`: 最大上下文长度，影响 KV cache 池大小
- `top_p / temperature`: 采样参数

## 项目结构

```
Genllm/
├── include/
│   ├── core/        # tensor, graph, executor, scheduler, manager, kernels
│   ├── backend/cpu/ # CPU 算子头文件
│   ├── backend/cuda/# CUDA 算子头文件
│   ├── model/       # ModelBase, Qwen3, OpFactory (图构建 DSL)
│   └── utils/       # bfloat16, float16, 类型定义
├── src/
│   ├── core/        # 核心实现 (GGUF 解析, 图, 执行器, 调度器, 分页注意力)
│   ├── backend/cpu/ # CPU 算子实现
│   ├── backend/cuda/# CUDA 算子实现 (.cu)
│   └── model/       # Qwen3 图构建, 模型工厂
├── tests/           # 测试入口
├── cmake/           # CMake 包配置
└── models/          # 模型文件 (.gguf)
```

## 当前限制

- 仅支持 batch=1 推理
- CUDA 后端需要 `CUDA_LAUNCH_BLOCKING=1` 或 per-tensor `cudaDeviceSynchronize()` 保证激活池复用正确性
- 仅实现 Qwen3 架构（模型工厂可扩展）

## License

MIT
