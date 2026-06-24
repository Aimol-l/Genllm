# CPU-only benchmark and optimization notes

本文记录一次基于 Qwen3-0.6B IQ4_NL GGUF 权重的 CPU-only 对比测试，并把结果转化为 Genllm 后续优化建议。对比对象是 llama.cpp，目标不是证明绝对性能，而是建立一个可以复测的 CPU 后端基线，定位当前 Genllm 的主要性能瓶颈。

## 测试环境

- CPU: Intel(R) Core(TM) Ultra 7 265K
- CPU cores/threads: 20 physical cores / 20 logical threads
- System memory: about 47.4 GB
- Model: `Qwen3-0.6B-IQ4_NL.gguf`
- Model type: Qwen3 0.6B, IQ4_NL, about 596M parameters
- Genllm build: CPU-only release build, `BACKEND_CPU=ON`
- llama.cpp mode: CPU backend only, `-ngl 0`

## 复测命令

llama.cpp:

```powershell
llama-bench.exe `
  -m Qwen3-0.6B-IQ4_NL.gguf `
  -ngl 0 -t 20 -p 512 -n 128 -r 3 -o csv

llama-bench.exe `
  -m Qwen3-0.6B-IQ4_NL.gguf `
  -ngl 0 -t 20 -p 12,13,18,20 -n 32 -r 3 -o csv
```

Genllm:

```powershell
cmake --build --preset cpu-release

benchmark.exe Qwen3-0.6B-IQ4_NL.gguf 32
```

## 对比结果

### llama.cpp CPU-only

`llama-bench --list-devices` 没有发现 GPU 设备，实际加载的是 CPU backend。测试中使用 `n_gpu_layers=0`。

| 阶段 | 配置 | 结果 |
|---|---:|---:|
| prompt processing | `n_prompt=512` | about 1346.8 tok/s |
| decode | `n_gen=128` | about 107.0 tok/s |
| short prompt processing | `n_prompt=12/13/18/20` | about 189.8 / 239.0 / 270.9 / 348.4 tok/s |
| short decode | `n_gen=32` | about 31.0 tok/s |

### Genllm CPU-only

Genllm 识别到 CPU 设备，并把全部层分配到 CPU：

- Device assignment: `CPU : L:-1 ~ L:28`
- Weight memory: about 358.2 MB used
- Activation pool: about 627.4 MB reserved
- KV cache: about 224.0 MB used / 235.2 MB reserved

`benchmark.exe` 生成 32 token 的端到端结果：

| Prompt | Prompt tokens | Generated tokens | Elapsed ms | ms/token | tokens/s |
|---|---:|---:|---:|---:|---:|
| `1+1=` | 12 | 32 | 11188.7 | 349.65 | 2.9 |
| `中国首都是哪里？` | 13 | 32 | 10084.4 | 315.14 | 3.2 |
| `Write a Python function to check if a number is prime.` | 20 | 32 | 13217.6 | 413.05 | 2.4 |
| `Translate into Chinese: Hello, how are you?` | 18 | 32 | 13310.4 | 415.95 | 2.4 |

## 结论

在同一模型、同一 CPU-only 条件下：

- llama.cpp 的 32-token decode 约为 31 tok/s。
- Genllm 的 32-token 端到端生成约为 2.4-3.2 tok/s。
- 以短 decode 结果估算，Genllm 当前 CPU 路径约慢 10-13 倍。
- 如果参考 llama.cpp `n_gen=128` 的 sustained decode 结果，和成熟 CPU 后端的差距可能超过 30 倍。

这说明 Genllm 的 GGUF 解析、图构建、CPU 调度、KV cache 和完整 decode 链路已经可运行；当前主要问题不是功能缺失，而是 CPU 内核和 benchmark 口径还没有达到成熟推理库的水平。

## 当前瓶颈分析

### 1. Quantized linear/GEMV 是最大瓶颈

Qwen3 decode 阶段大部分计算来自线性层。Genllm 当前 CPU `linear` 对 `IQ4_NL` 权重的典型路径是：

1. 按输出列块读取量化权重。
2. 先反量化到临时 `bfloat16_t` buffer。
3. 再执行 BF16 -> FP32 的 dot product。

这条路径功能清晰，但对 decode 不够高效：

- 反量化结果被写入临时 BF16 buffer，增加内存流量。
- decode 通常是 `M=1` GEMV，通用矩阵分块无法充分摊薄调度和反量化成本。
- 每个输出块重复执行 dequantize + BF16 load + BF16 to FP32 转换。

llama.cpp 的 CPU 后端更接近 fused quantized GEMV：量化 nibble 解码、scale 应用、向量化乘加在同一内核中完成，尽量避免中间 BF16 写回。

### 2. CPU attention 仍是标量路径

当前 paged attention 的 CPU 路径主要是 head/token/dim 的标量循环，并且存在 per-head scratch vector 创建。短上下文下 linear 更重；上下文变长后，attention 的标量 dot、softmax 和 V 累加会逐渐成为明显瓶颈。

### 3. Sampling 会污染小模型 benchmark

当前 sampling 需要对完整 vocab 做 softmax，top-p 路径还会对全 vocab 排序。Qwen3-0.6B 的 vocab 约 151936，对小模型 CPU decode 来说这部分成本已经可见。

如果 benchmark 目的是评估后端算子，应提供 greedy/argmax 或固定 token 模式，把 sampling 单独计时。

### 4. Benchmark 口径还没有拆分

现有 benchmark 使用 `Executor::generate()` 端到端计时，会混合：

- prompt prefill
- per-token decode
- sampling
- tokenizer prompt preparation
- 首次执行路径开销

这适合观察整体可用性，但不足以指导内核优化。下一步需要拆分 prefill、decode、sampling 和 per-op / per-layer 计时。

### 5. KV cache pool 余量偏紧

本次 `max_seq_len=2048` 下 KV cache 使用约 224.0 MB / 235.2 MB，使用率约 95.2%。这说明当前配置可跑通，但对更长上下文、batch 或多设备切分不够宽松。

## 参考 llama.cpp 的优化顺序

### P0: 先补齐可复测 benchmark

优先做 benchmark，否则后续优化很难判断收益是否真实。

建议新增：

- `bench_prefill`: 单独记录 prompt processing tok/s。
- `bench_decode`: 预填 prompt 后，只统计 decode tok/s。
- `bench_sampling`: 单独记录 softmax、top-p、argmax 成本。
- `bench_linear_iq4nl`: 单独压测 `IQ4_NL x BF16/FP32` GEMV。
- per-layer/per-op timer: 记录 linear、attention、rms_norm、rope、silu、elementwise 的耗时占比。

输出建议固定为 CSV 或 JSONL，方便后续每次 PR 做回归对比。

### P1: 为 IQ4_NL 实现 fused CPU GEMV

这是最可能带来数量级收益的优化。

建议方向：

- 为 `M=1` decode 单独写 `IQ4_NL` GEMV kernel。
- 避免把整块权重反量化到 BF16 buffer。
- 直接从 4-bit nibble 查 `IQ4_NL` value table，乘 block scale 后进入 FP32 accumulator。
- 对 hidden/intermediate/vocab projection 的常见 shape 做专用 fast path。
- 按输出列分块并行时减少 OpenMP 小任务开销。
- 先以单层 micro-benchmark 验证，再接入整图。

可以把目标拆成两步：

1. direct dequantize-and-dot scalar/reference kernel，保证正确性。
2. AVX2/FMA optimized kernel，追求吞吐。

### P2: 优化 logits projection 和 sampling

Qwen3 vocab 较大，最后 logits projection 和 sampling 对小模型很重。

建议：

- benchmark 增加 greedy/argmax 模式。
- top-p 不做全量 sort，改为 top-k/partial selection 后再累计。
- 研究 logits projection + top-k 融合，避免总是完整 materialize 全 vocab logits 后再排序。

### P3: 优化 CPU paged attention

建议：

- 复用 per-thread scratch buffer，避免每个 head 创建临时 vector。
- 为 head_dim=128 写 AVX2 dot 和 V accumulation。
- 检查 KV cache 内存布局，尽量让 decode 读 K/V 连续。
- 为 single-token decode 写专用 attention path。

### P4: 减少 executor 和 runtime dispatch 开销

建议：

- 固化 decode 阶段不会变化的 tensor shape 和 op metadata。
- 检查每 token 是否有重复 `resolve_dims`、activation reset、临时分配。
- 对热路径 dtype dispatch 做轻量化处理。
- 加入 allocator/debug 计数，定位每 token 分配次数。

### P5: 之后再统一比较 CPU/Vulkan/CUDA 后端

CPU-only benchmark 建立后，再用相同 prompt、token 数、采样策略比较 Vulkan/CUDA。这样可以区分：

- 算子内核问题
- 调度/内存池问题
- GPU 同步问题
- sampling/tokenizer 问题

## 建议的近期任务拆分

1. Add `bench_decode.cpp` and CSV/JSONL output.
2. Add `bench_linear_iq4nl.cpp` for isolated quantized GEMV testing.
3. Implement reference direct IQ4_NL dequantize-and-dot GEMV.
4. Optimize the IQ4_NL GEMV with AVX2/FMA.
5. Replace full vocab top-p sort with partial selection in benchmark mode.
6. Add CPU paged attention scratch-buffer reuse.

## PR 范围说明

本文档只记录 benchmark 结果和优化路线，不修改推理实现。后续实现 PR 应以 benchmark 为先导，逐项验证收益，避免一次性混合多个优化点导致性能归因不清。
