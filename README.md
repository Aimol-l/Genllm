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
| Qwen3-0.6B | 0.6B | 28 | 1024 | 16/8 | ✅ CPU + CUDA + Vulkan |
| Qwen3-1.7B | 1.7B | 28 | 2048 | 16/8 | ✅ CPU + CUDA + Vulkan |
| Qwen3-4B | 4B | 36 | 2560 | 32/8 | ✅ CPU + CUDA + Vulkan |

## 后端

| 后端 | 状态 | 已实现算子 |
|------|------|-----------|
| CPU | ✅ 完成 | linear, rms_norm, add/mul, silu, apply_rope, softmax, embedding, attention, reshape, permute, memcpy |
| CUDA | ✅ 完成 | 同上全部 + paged/flash attention (cuBLAS GEMM) |
| Vulkan | ✅ 完成 | 同上全部 + paged/flash attention (GLSL compute shaders) |
| SYCL | 🔜 计划 | — |

## 编译器要求

| 工具 | 版本要求 | 用途 | 必需 |
|------|---------|------|:----:|
| C++ 编译器 | C++23 (GCC 15+ / Clang 20+) | 核心引擎编译 | ✅ 是 |
| CMake | ≥ 3.28 | 构建系统 | ✅ 是 |
| Ninja / Make | — | 构建工具 | ✅ 是 |
| CUDA Toolkit | ≥ 12.8 (nvcc) | CUDA 后端编译 | CUDA 选 |
| Vulkan SDK | ≥ 1.4 (glslc + libvulkan) | Vulkan 后端 + GLSL→SPIR-V 编译 | Vulkan 选 |
| Python 3 | ≥ 3.8 | 统一构建脚本 (`build.py`)、着色器自动编译 | ✅ 是 |

**CUDA**: `nvcc` 编译 `.cu` 源文件，依赖 `cuBLAS`（CUDA Toolkit 自带）。  
**Vulkan**: `glslc` 将 `.comp` 着色器编译为 SPIR-V，`build.py --shader` 自动调用并嵌入 C++ header。
## 构建统一脚本

```bash
python build.py                        # CPU only (默认)
python build.py --vulkan               # CPU + Vulkan
python build.py --cuda                 # CPU + CUDA
python build.py --sycl                 # CPU + SYCL
python build.py --vulkan --test -j8    # Vulkan + 测试 + 8 线程
python build.py --shader               # 仅编译 GLSL shader → SPIR-V → C++ header,自动查找 ./shader/*/*.comp
python build.py --rebuild              # 清除 + 全量重建
python build.py --clean                # 清除构建产物
```

## Python构建
```
# CPU
python build.py --cpu --test -j$(nproc)

# CPU + CUDA
python build.py --cuda --test -j$(nproc)

# CPU + Vulkan
python build.py --vulkan --test -j$(nproc)  # 会自动编译 GLSL shaders

```

## CMake 手动构建

```bash
# CPU only
cmake -B build -DBACKEND_CPU=ON -DBUILD_TEST=ON
cmake --build build -j$(nproc)

# CPU + CUDA (需要 CUDA Toolkit 13+)
cmake -B build -DBACKEND_CUDA=ON -DBUILD_TEST=ON
cmake --build build -j$(nproc)

# CPU + Vulkan
# 需要先使用: python build.py --shader 得到spv文件并嵌入到代码中
cmake -B build -DBACKEND_VULKAN=ON -DBUILD_TEST=ON
cmake --build build -j$(nproc)
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
│   ├── add/               # 逐元素加
│   ├── sub/               # 逐元素减
│   ├── mul/               # 逐元素乘
│   ├── div/               # 逐元素除
│   ├── linear/            # 矩阵乘法 (cooperative matrix + warp-level GEMV)
│   ├── rms_norm/          # RMS 层归一化
│   ├── layer_norm/        # Layer 层归一化
│   ├── silu/              # SiLU 激活
│   ├── gelu/              # GELU 激活
│   ├── relu/              # ReLU 激活
│   ├── rope/              # RoPE 位置编码
│   ├── permute/           # 维度重排
│   ├── attention/         # 缩放点积注意力 + 分页注意力 FlashAttention
├── build.py               # 统一构建脚本
├── tests/                 # 测试入口
├── cmake/                 # CMake 包配置
└── models/                # 模型文件 (.gguf)
```

## 性能

Vulkan 后端在 decode 阶段使用 warp-level GEMV（非 cooperative matrix），
在 RTX 4070 Ti SUPER 上的 benchmark 结果：

| 模型 | 后端 | tokens/s |
|------|------|:--------:|
| Qwen3-0.6B | Vulkan | ~44 |
| Qwen3-4B   | Vulkan | ~24 |

## 当前限制

- 仅支持 batch=1 推理
- CUDA 后端需要 `CUDA_LAUNCH_BLOCKING=1` 或 per-tensor `cudaDeviceSynchronize()` 保证激活池复用正确性
- 仅实现 Qwen3/Qwen3.5 架构（模型工厂可扩展）

## License

MIT
