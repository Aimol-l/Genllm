# 添加新算子指南（无宏版本）

## 示例：添加 GELU 激活函数

### 步骤1：在 OperationType 枚举中添加

```cpp
// utils/utils.hpp
enum class OperationType : uint32_t {
    // ... 现有算子
    OP_TYPE_GELU = 100,  // 新增
};
```

### 步骤2：实现 CPU 版本

```cpp
// src/cpu/cpu_gelu.h
#pragma once
#include "core/operator.h"
#include "tensor.hpp"

namespace cpu {

// GELU 实现
inline void gelu_forward(Tensor* output, const Tensor* input) {
    size_t n = 1;
    for (int i = 0; i < TENSOR_MAX_DIMS && input->dims[i] != 0; ++i) {
        n *= input->dims[i];
    }

    const float* in = static_cast<const float*>(input->data);
    float* out = static_cast<float*>(output->data);

    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        // GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float t = std::tanh(0.7978845608f * (x + 0.044715f * x * x * x));
        out[i] = 0.5f * x * (1.0f + t);
    }
}

} // namespace cpu
```

### 步骤3：注册 CPU 算子

```cpp
// src/cpu/cpu_ops.h (添加到文件末尾)

// 首先添加注册函数声明到 operator.h:
inline void register_cpu_gelu(OpKernelFunction func);

// 然后实现内核包装
namespace {
    inline void cpu_gelu_kernel(Tensor* tensor, const OpContext& ctx) {
        if (!tensor->src[0]) {
            throw std::runtime_error("CPU GELU: missing input");
        }
        cpu::gelu_forward(tensor, tensor->src[0]);
    }
}

// 最后在 init_cpu_kernels() 中注册
inline void init_cpu_kernels() {
    // ... 现有注册
    register_cpu_gelu(cpu_gelu_kernel);  // 新增
}
```

### 步骤4：实现 CUDA 版本（可选）

```cpp
// src/cuda/cuda_gelu.h
#if GENLLM_HAS_CUDA

namespace cuda {

__global__ void gelu_kernel(float* output, const float* input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
        output[idx] = 0.5f * x * (1.0f + t);
    }
}

inline void gelu_forward(Tensor* output, const Tensor* input, cudaStream_t stream = 0) {
    size_t n = /* 计算元素数量 */;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    gelu_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<float*>(output->data),
        static_cast<const float*>(input->data),
        n
    );
}

} // namespace cuda

// 注册
namespace {
    inline void cuda_gelu_kernel(Tensor* tensor, const OpContext& ctx) {
        if (!tensor->src[0]) {
            throw std::runtime_error("CUDA GELU: missing input");
        }
        cudaStream_t stream = ctx.stream ? static_cast<cudaStream_t>(ctx.stream) : 0;
        cuda::gelu_forward(tensor, tensor->src[0], stream);
    }
}

inline void init_cuda_kernels() {
    // ... 现有注册
    register_cuda_gelu(cuda_gelu_kernel);  // 新增
}

#endif // GENLLM_HAS_CUDA
```

### 步骤5：在 helper 函数中添加便捷接口

```cpp
// tensor.hpp
inline Tensor* gelu(Tensor* x, const std::string& name = "") {
    if (!x) return nullptr;

    std::vector<int64_t> out_dims;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) {
        out_dims.push_back(x->dims[i]);
    }

    auto* out = Tensor::create_new_tensor(x->dtype, out_dims, name, OperationType::OP_TYPE_GELU);
    out->src[0] = x;
    return out;
}
```

### 步骤6：使用

```cpp
// 构建计算图
Tensor* x = create_tensor(...);
Tensor* y = gelu(x, "gelu_out");  // 创建 GELU 节点

// 执行时会自动选择 CPU 或 CUDA 实现
dispatcher.dispatch(y);
```

## 完整示例：添加矩阵乘法

### 1. 定义算子类型
```cpp
// 已经存在: OP_TYPE_MUL_MAT
```

### 2. CPU 实现
```cpp
// src/cpu/cpu_matmul.h
namespace cpu {

template<typename T>
void matmul_forward(Tensor* output, const Tensor* a, const Tensor* b) {
    // 简化实现：假设 [M, K] * [K, N] = [M, N]
    int64_t M = a->dims[0];
    int64_t K = a->dims[1];
    int64_t N = b->dims[1];

    const T* a_ptr = static_cast<const T*>(a->data);
    const T* b_ptr = static_cast<const T*>(b->data);
    T* out_ptr = static_cast<T*>(output->data);

    // 朴素实现（实际应该使用优化的 BLAS）
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            T sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_ptr[m * K + k] * b_ptr[k * N + n];
            }
            out_ptr[m * N + n] = sum;
        }
    }
}

} // namespace cpu
```

### 3. 注册
```cpp
// operator.h 添加
inline void register_cpu_mul_mat(OpKernelFunction func);

// cpu_ops.h 添加
namespace {
    inline void cpu_mul_mat_kernel(Tensor* tensor, const OpContext& ctx) {
        if (!tensor->src[0] || !tensor->src[1]) {
            throw std::runtime_error("CPU MatMul: missing inputs");
        }
        // 根据数据类型调用不同模板
        cpu::matmul_forward<float>(tensor, tensor->src[0], tensor->src[1]);
    }
}

inline void init_cpu_kernels() {
    // ...
    register_cpu_mul_mat(cpu_mul_mat_kernel);
}
```

## 优势对比

| 特性 | 宏版本 | 无宏版本 |
|------|--------|----------|
| 可读性 | ❌ 隐式魔法 | ✅ 显式调用 |
| 调试 | ❌ 宏展开难追踪 | ✅ 直接断点 |
| 类型安全 | ❌ 字符串拼接 | ✅ 编译时检查 |
| IDE 支持 | ❌ 自动补全差 | ✅ 完整支持 |
| 学习曲线 | ❌ 需要理解宏 | ✅ 直观易懂 |

## 总结

**添加新算子的步骤：**

1. 在 `utils/utils.hpp` 添加 `OperationType` 枚举值
2. 在对应后端实现算子函数（如 `cpu::xxx_forward`）
3. 在 `operator.h` 添加注册函数声明（如 `register_cpu_xxx`）
4. 创建内核包装函数（如 `cpu_xxx_kernel`）
5. 在 `init_cpu_kernels()` 调用注册函数
6. （可选）在 `tensor.hpp` 添加便捷创建函数

**无需任何宏！**
