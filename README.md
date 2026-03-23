# GGUF Parser

一个高性能的 C++ GGUF 文件解析器，用于读取和分析 GPT-Generated Unified Format (GGUF) 模型文件。

## 特性

- 完整解析 GGUF 文件格式（支持 v3+）
- 支持所有 GGML 数据类型（F32, F16, BF16, Q4_0, Q4_1, Q8_0 等）
- 自动解量化到 float
- 提取模型元数据和超参数
- 跨平台支持（Linux, macOS, Windows）

## 构建方法

### 要求

- CMake 3.15+
- C++17 编译器（GCC 8+, Clang 10+, MSVC 2019+）

### 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 安装

```bash
make install
```

## 使用方法

### gguf_tool - 主要工具

```bash
# 基本用法
./gguf_tool <model.gguf>

# 显示特定张量数据
./gguf_tool model.gguf --tensor token_embd.weight

# 完整模式（显示所有张量）
./gguf_tool model.gguf --inspect

# 显示所有元数据
./gguf_tool model.gguf --metadata
```

### gguf_inspect - 快速检查工具

```bash
# 基本摘要
./gguf_inspect model.gguf

# 显示统计信息
./gguf_inspect model.gguf --stats

# 按层分析
./gguf_inspect model.gguf --layers
```

## 示例输出

```
========================================
         GGUF File Information
========================================
Version:      3
Tensor Count: 293
KV Count:     284
Alignment:    32 bytes
Data Offset:  18432 (0x4800)

========================================
         Model Architecture
========================================
Architecture: qwen2
Hidden Size:     1536
Num Layers:      24
Num Heads:       12
KV Heads:        2
Vocab Size:      151936

========================================
         Tensors (first 10 of 293)
========================================
  token_embd.weight               | shape: [151936, 1536] | type: F16   | offset: 0 | size: 438.53 MB
  blk.0.attn_q.weight             | shape: [1536, 1536]   | type: Q4_0  | offset: 459770880 | size: 12.00 MB
  ...
```

## 代码示例

```cpp
#include "gguf_parser.h"

int main() {
    gguf::GGUFParser parser("model.gguf");
    auto info = parser.parse();

    // 获取模型参数
    std::string arch = info.get_string("general.architecture");
    int64_t hidden_size = info.get_int("llama.embedding_length");

    // 读取张量数据
    for (const auto& tensor : info.tensors) {
        if (tensor.name == "token_embd.weight") {
            auto data = parser.read_tensor_float(info, tensor);
            // 使用数据...
        }
    }

    return 0;
}
```

## GGUF 文件格式

```
┌─────────────────────────────────────────────────────┐
│  Header (12 字节)                                    │
│  - magic: "GGUF" (4 字节)                            │
│  - version: uint32 (4 字节)                          │
│  - tensor_count: uint64 (8 字节)                     │
│  - kv_count: uint64 (8 字节)                         │
├─────────────────────────────────────────────────────┤
│  KV 元数据区 (kv_count 个键值对)                      │
│  - key: string (长度 + 内容)                         │
│  - value: 类型标记 + 数据                            │
├─────────────────────────────────────────────────────┤
│  张量信息区 (tensor_count 个张量)                     │
│  - name: string                                      │
│  - n_dims: uint32                                    │
│  - dimensions: uint64[n_dims]                        │
│  - dtype: uint32                                     │
│  - offset: uint64 (相对于数据区的偏移)                │
├─────────────────────────────────────────────────────┤
│  对齐填充 (到 32 字节边界)                             │
├─────────────────────────────────────────────────────┤
│  张量数据区 (所有张量的原始二进制数据)                 │
└─────────────────────────────────────────────────────┘
```

## 支持的数据类型

| 类型   | 描述            | 大小  |
|--------|---------------|-------|
| F32    | 32位浮点         | 4 B   |
| F16    | 16位浮点         | 2 B   |
| BF16   | bfloat16       | 2 B   |
| Q4_0   | 4位量化（块大小32）| 变长  |
| Q4_1   | 4位量化 + 偏移   | 变长  |
| Q8_0   | 8位量化         | 变长  |
| Q2_K   | K-quants 2位   | 变长  |
| Q3_K   | K-quants 3位   | 变长  |
| Q4_K   | K-quants 4位   | 变长  |
| Q5_K   | K-quants 5位   | 变长  |
| Q6_K   | K-quants 6位   | 变长  |

## License

MIT
