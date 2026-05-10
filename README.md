# Genllm

基于 C++23 的模块化 LLM 推理框架，支持 CPU / CUDA / Vulkan 后端，原生解析 GGUF 格式。

## 特性

- **GGUF 解析** — 完整的 GGUF v3 反序列化，支持 BPE tokenizer、BF16/F16/F32 数据类型
- **计算图 IR** — 基于 DAG 的可视化计算图，拓扑排序执行，支持 DOT 导出
- **分页注意力** — 分块 KV cache（PAGE_BLOCK_SIZE=16），支持多设备按层分配
- **双层内存池** — 权重/激活/KV cache 三池分离，激活池层间复用
- **自动调度** — 按显存容量自动分配连续层到 CPU/GPU 设备，跨设备自动插入 copy 边
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
| Vulkan | 🚧 开发中 | add/sub/mul/div (f16/f32/bf16) |
| SYCL | 🔜 计划 | — |

## 构建统一脚本

```bash
python build.py                        # CPU only (默认)
python build.py --vulkan               # CPU + Vulkan
python build.py --cuda                 # CPU + CUDA
python build.py --sycl                 # CPU + SYCL
python build.py --vulkan --test -j8    # Vulkan + 测试 + 8 线程
python build.py --shader               # 仅编译 GLSL shader → SPIR-V → C++ header
python build.py --rebuild              # 清除 + 全量重建
python build.py --clean                # 清除构建产物
```

## CMake 手动构建

```bash
# CPU only
cmake -B build -DBACKEND_CPU=ON -DBUILD_TEST=ON
cmake --build build -j

# CPU + CUDA (需要 CUDA Toolkit 13+)
cmake -B build -DBACKEND_CPU=ON -DBACKEND_CUDA=ON -DBUILD_TEST=ON
cmake --build build -j

# CPU + Vulkan
cmake -B build -DBACKEND_CPU=ON -DBACKEND_VULKAN=ON -DBUILD_TEST=ON
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
│   ├── core/              # tensor, graph, executor, scheduler, manager, kernels
│   ├── backend/cpu/       # CPU 算子头文件
│   ├── backend/cuda/      # CUDA 算子头文件
│   ├── backend/vulkan/    # Vulkan 算子头文件 + SPIR-V 嵌入 (spv/)
│   ├── model/             # ModelBase, Qwen3, OpFactory (图构建 DSL)
│   └── utils/             # bfloat16, float16, 类型定义
├── src/
│   ├── core/              # 核心实现 (GGUF 解析, 图, 执行器, 调度器, 分页注意力)
│   ├── backend/cpu/       # CPU 算子实现
│   ├── backend/cuda/      # CUDA 算子实现 (.cu)
│   ├── backend/vulkan/    # Vulkan 算子实现
│   └── model/             # Qwen3 图构建, 模型工厂
├── shader/                # GLSL compute shaders
│   ├── add/               # add_f32.comp, add_bf16.comp, add_f16.comp
│   ├── sub/               # ...
│   ├── mul/
│   └── div/
├── build.py               # 统一构建脚本
├── tests/                 # 测试入口
├── cmake/                 # CMake 包配置
└── models/                # 模型文件 (.gguf)
```

## 当前限制

- 仅支持 batch=1 推理
- CUDA 后端需要 `CUDA_LAUNCH_BLOCKING=1` 或 per-tensor `cudaDeviceSynchronize()` 保证激活池复用正确性
- Vulkan 后端仅有算子骨架，大部分算子尚未实现
- 仅实现 Qwen3 架构（模型工厂可扩展）

## License

MIT
