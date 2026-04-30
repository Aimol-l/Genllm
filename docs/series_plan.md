# 系列名称

**「从零手写 LLM 推理引擎」**

英文副标题：Building an LLM Inference Engine from Scratch in C++

---

# 视频规划

## 第 1 集：项目总览 & GGUF 解析
- 整体架构介绍：计算图、调度器、执行器、后端的四层结构
- GGUF 文件格式：元数据 KV、张量信息表、数据类型映射
- C++ GGUF 解析器完整实现，支持 BF16/F16/F32/量化类型
- BPE Tokenizer 实现

## 第 2 集：计算图 IR
- 为什么需要计算图：对比手写逐层 forward vs 图执行
- Tensor 设计：dims、strides、src 依赖、op_params
- OpFactory DSL：linear、rms_norm、silu、add、reshape、permute
- 图构建：从 output 反向 BFS 收集 + 拓扑排序
- DOT 导出可视化

## 第 3 集：Qwen3 模型结构
- 模型架构介绍：RoPE、GQA、SwiGLU、RMSNorm、QK-Norm
- 用 OpFactory 搭出完整的 Qwen3 Transformer
- 对比 0.6B / 1.7B / 4B 的参数差异

## 第 4 集：自动调度 & 内存管理
- 层级成本估算：weight + activation + kv_cache
- 连续层分配算法（类似 pipeline parallelism）
- MemoryPool 设计：bump allocator + 层间激活复用
- 跨设备 copy edge 自动插入
- 多设备池化：weight 池 / activation 池 / kv_cache 池

## 第 5 集：CPU 推理核心
- 线性层优化：BF16 GEMM 分块（MR/NR/KR tiling）+ AVX2
- RMSNorm：warp-level 单行并行
- RoPE：预计算 cos/sin 表 + 原位旋转
- embedding、add、silu 等 element-wise 算子
- 对比手写和 OpenMP 的优化收益

## 第 6 集：分页注意力 & KV Cache
- KV cache 原理：免重算加速 decode
- PagedAttention 设计：16 token 分块，动态 page table
- 写扩散问题：append_kv_from_tensor 的 layout
- 在线 softmax：safe softmax + rescale 技巧
- 对比有无 KV cache 的推理速度

## 第 7 集：自回归生成 & 采样
- prefill + decode 双阶段执行
- RoPE start_pos 的逐 token 推进
- 采样算法：argmax、top-p、temperature
- 流式输出：边生成边 decode
- 完整的 generate() 执行流程

## 第 8 集：CUDA 推理实现（上）—— 基础算子
- CUDA kernel 基础：grid-stride loop、warp reduce、shared memory
- RMSNorm CUDA：每个 warp 处理一行
- 激活函数：silu/gelu/relu 的 element-wise kernel
- cv::Softmax：warp-level online softmax
- 调试工具：cuda-memcheck、nsys、CUDA_LAUNCH_BLOCKING

## 第 9 集：CUDA 推理实现（中）—— cuBLAS & 性能
- cuBLAS 集成：handle 池化、GEMM API
- row-major vs column-major 的混淆与避坑
- 常见 bug：permute kernel 的 stride 计算错误
- 性能对比：CPU vs CUDA 逐层耗时
- cuBLAS handle 复用 vs 每次创建的性能差异

## 第 10 集：CUDA 推理实现（下）—— 分页注意力 & 全流程贯通
- CUDA paged attention kernel：page table + block pool
- append KV 到 cache 的 device-aware memcpy
- 跨 CUDA 设备拷贝（cudaMemcpy 自动路由）
- 踩坑实录：激活池竞态、QK-Norm 维度错误、permute 偏移
- CPU → CUDA 迁移的数据流（copy edge）

## 第 11 集：多后端架构 & 可扩展设计
- 设备枚举 + dispatchOp 模板化后端分发
- 添加新模：ModelFactory 注册机制
- 算子扩展流程：从 OpFactory 到 kernel dispatch
- SYCL / Vulkan 的接口预留

## 第 12 集：项目总结 & 经验分享
- 整体架构回顾
- 开发过程中踩过的坑（GQA hardcode、permute 偏移、同步竞态）
- C++23 在生产项目中的体验（std::println、flat_map、format）
- 性能数据和优化方向
- 未来规划：FlashAttention、量化推理、continuous batching
