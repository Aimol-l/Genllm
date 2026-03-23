# 计算图执行流程说明

## 核心概念

**执行顺序 ≠ 子图顺序**

- ✅ **执行顺序**：由数据依赖关系决定（拓扑排序）
- ✅ **子图作用**：决定节点在哪个设备上执行、使用哪个内存池

## 示例：Qwen3 Transformer 层

### 计算图结构

```
                    input_ids (CPU)
                         ↓
                    token_embd (CPU)
                         ↓
                    hidden_states (CPU)
                         ↓
        ┌────────────────┴────────────────┐
        │                                 │
    attn_norm (CPU)                residual (保存)
        │                                 │
        ↓                                 │
    Q_proj (GPU0)                         │
    K_proj (GPU1)                         │
    V_proj (GPU1)  ← 可并行执行           │
        │                                 │
        └────────────────┬────────────────┘
                         ↓
                 attention (GPU0)
                         ↓
                 attn_output (GPU0)
                         ↓
                add (CPU) ← attn_out + residual
                         ↓
                    ffn_norm (CPU)
                         ↓
                 gate_proj (GPU1)
                 up_proj (GPU1)
                         ↓
                   swiglu (GPU1)
                         ↓
                down_proj (GPU1)
                         ↓
                  ffn_out (GPU1)
                         ↓
                add (CPU) ← ffn_out + input
                         ↓
                   output (CPU)
```

### 子图划分（按设备）

```
子图0 (CPU):
  - token_embd
  - attn_norm
  - residual connections (add)
  - ffn_norm
  - output

子图1 (GPU0):
  - Q_proj
  - attention
  - attn_output

子图2 (GPU1):
  - K_proj
  - V_proj
  - gate_proj
  - up_proj
  - swiglu
  - down_proj
```

### 执行顺序（拓扑序）

```python
# 按数据依赖顺序执行，不是按子图顺序！

Step 1:  [CPU]    token_embd        # 独立执行
Step 2:  [CPU]    attn_norm         # 依赖 Step 1
Step 3:  [GPU0]   Q_proj            # 依赖 Step 2
Step 4:  [GPU1]   K_proj            # 依赖 Step 2，与 Step 3 并行
Step 5:  [GPU1]   V_proj            # 依赖 Step 2，与 Step 3,4 并行
Step 6:  [GPU0]   attention         # 依赖 Step 3,4,5
Step 7:  [GPU0]   attn_output       # 依赖 Step 6
Step 8:  [CPU]    add_attn          # 依赖 Step 7
Step 9:  [CPU]    ffn_norm          # 依赖 Step 8
Step 10: [GPU1]   gate_proj         # 依赖 Step 9
Step 11: [GPU1]   up_proj           # 依赖 Step 9，与 Step 10 并行
Step 12: [GPU1]   swiglu            # 依赖 Step 10,11
Step 13: [GPU1]   down_proj         # 依赖 Step 12
Step 14: [CPU]    ffn_out           # 依赖 Step 13
Step 15: [CPU]    add_ffn           # 依赖 Step 14
```

## 关键点

### 1. 数据依赖决定执行顺序
```
必须等所有输入就绪才能执行：
- attention 等待 Q, K, V 都计算完
- add 等待两个输入都计算完
```

### 2. 子图决定执行设备
```
- 子图0 → CPU 设备 → CPU 内存池
- 子图1 → GPU0 设备 → GPU0 内存池
- 子图2 → GPU1 设备 → GPU1 内存池
```

### 3. 并行执行机会
```
不同设备上无依赖的节点可以并行：
- Step 3,4,5 可以并行（GPU0 和 GPU1 同时工作）
- Step 10,11 可以并行（GPU1 上多个矩阵乘法）
```

### 4. 内存管理
```
每个子图有独立的内存池：
- 子图0 的临时张量在 CPU 内存池分配
- 子图1 的临时张量在 GPU0 内存池分配
- 子图2 的临时张量在 GPU1 内存池分配

跨子图数据传输：
- CPU → GPU0: Q_proj 需要从 CPU 拷贝输入到 GPU0
- GPU1 → GPU0: K,V 需要从 GPU1 拷贝到 GPU0
```

## 代码对应

### Scheduler::build_execute_order()
```cpp
// 拓扑排序生成执行顺序
void Scheduler::build_execute_order(Tensor* root) {
    // 按 BFS 生成拓扑序
    // 结果：[node1, node2, node3, ...]
    //      按数据依赖排序，不是按子图排序
}
```

### Executor::execute()
```cpp
void Executor::execute(Tensor* root) {
    for (const auto& task : execute_order) {
        // task.node          ← 要执行的节点
        // task.subgraph      ← 节点所属的子图
        // task.subgraph->device ← 在哪个设备上执行

        dispatch_operation(task.node);  // 执行
    }
}
```

## 总结

| 概念 | 由什么决定 | 作用 |
|------|-----------|------|
| **执行顺序** | 数据依赖关系（拓扑序） | 决定先算哪个节点 |
| **子图划分** | 设备类型、负载均衡 | 决定节点在哪执行 |
| **内存分配** | 子图/设备 | 决定使用哪个内存池 |
| **并行机会** | 设备数量 + 无依赖关系 | 决定哪些可以同时执行 |

**记住：拓扑序在前，设备在后！**
