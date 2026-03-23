# 算子分发系统设计说明

## 核心问题

**问题**：如何根据节点分配的后端（CPU/CUDA），自动调用对应的算子实现？

## 解决方案：算子注册表 + 分发器

### 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   算子分发流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 节点分配到子图                                          │
│     tensor.backend = CPU/GPU/Vulkan                         │
│                                                             │
│  2. 执行时遍历节点（拓扑序）                                 │
│     for (node : execute_order)                              │
│                                                             │
│  3. 根据 tensor.backend 选择对应的算子内核                   │
│     dispatcher.dispatch(tensor)                             │
│       ↓                                                     │
│     registry.get(backend, op_type)                          │
│       ↓                                                     │
│     kernel->execute(tensor)                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. **OpKernel** - 算子内核描述
```cpp
struct OpKernel {
    Backend backend;              // CPU, CUDA, Vulkan
    OperationType op_type;        // ADD, MUL_MAT, RELU, ...
    std::function<void(Tensor*, OpContext)> func;  // 执行函数
    bool is_available;
};
```

#### 2. **OpRegistry** - 算子注册表
```cpp
class OpRegistry {
    // 键: (backend, op_type) → 值: OpKernel
    std::unordered_map<uint64_t, OpKernel> m_kernels;

    void register_kernel(const OpKernel& kernel);
    const OpKernel* get_kernel(Backend backend, OperationType op_type);
};
```

#### 3. **OpDispatcher** - 算子分发器
```cpp
class OpDispatcher {
    void dispatch(Tensor* tensor, const OpContext& ctx) {
        // 1. 从 tensor->backend 获取设备类型
        // 2. 查找对应的内核
        // 3. 执行内核
    }
};
```

### 使用示例

#### 步骤1：实现 CPU 算子
```cpp
// cpu_ops.h
namespace cpu {
    void relu_forward(Tensor* output, const Tensor* input) {
        // CPU 实现（使用循环）
        for (size_t i = 0; i < n; ++i) {
            output[i] = std::max(0.0f, input[i]);
        }
    }
}
```

#### 步骤2：注册 CPU 算子
```cpp
REGISTER_BACKEND_KERNELS(Backend::CPU, {
    OpKernel(Backend::CPU, OP_TYPE_RELU, "relu",
        [](Tensor* t, const OpContext& ctx) {
            cpu::relu_forward(t, t->src[0]);
        }),
    // ... 其他算子
});
```

#### 步骤3：实现 CUDA 算子
```cpp
// cuda_ops.h
namespace cuda {
    __global__ void relu_kernel(float* output, const float* input, size_t n) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = fmaxf(0.0f, input[idx]);
        }
    }

    void relu_forward(Tensor* output, const Tensor* input) {
        // CUDA 实现（调用 kernel）
        relu_kernel<<<grid, block>>>(...);
    }
}
```

#### 步骤4：注册 CUDA 算子
```cpp
REGISTER_BACKEND_KERNELS(Backend::CUDA, {
    OpKernel(Backend::CUDA, OP_TYPE_RELU, "relu",
        [](Tensor* t, const OpContext& ctx) {
            cuda::relu_forward(t, t->src[0]);
        }),
    // ... 其他算子
});
```

#### 步骤5：执行时自动分发
```cpp
// executor.cpp
void Executor::execute_node(Tensor* node) {
    // 自动根据 node->backend 选择对应的内核
    OpDispatcher& dispatcher = get_op_dispatcher();
    dispatcher.dispatch(node);  // CPU→cpu::relu, CUDA→cuda::relu
}
```

## 完整流程示例

### 场景：执行 ReLU

```cpp
// 1. 图构建时
Tensor* relu1 = Tensor::create_new_tensor(F32, {1024}, "relu1", OP_TYPE_RELU);
relu1->src[0] = some_input_tensor;
relu1->backend = cpu_resource;  // 分配到 CPU

Tensor* relu2 = Tensor::create_new_tensor(F32, {1024}, "relu2", OP_TYPE_RELU);
relu2->src[0] = some_input_tensor;
relu2->backend = cuda_resource;  // 分配到 CUDA

// 2. 图切分时
subgraph0.nodes.push_back(relu1);  // CPU 子图
subgraph1.nodes.push_back(relu2);  // CUDA 子图

// 3. 执行时（按拓扑序）
for (const auto& task : execute_order) {
    // task.node 可能是 relu1 或 relu2

    // 分发器根据 tensor->backend 自动选择内核
    dispatcher.dispatch(task.node);

    // 如果是 relu1 (CPU)   → 调用 cpu::relu_forward()
    // 如果是 relu2 (CUDA)  → 调用 cuda::relu_forward()
}
```

## 关键设计点

### ✅ 1. 编译时注册
```cpp
// 全局静态变量自动注册
REGISTER_BACKEND_KERNELS(Backend::CPU, get_cpu_kernels());
REGISTER_BACKEND_KERNELS(Backend::CUDA, get_cuda_kernels());
```

### ✅ 2. 运行时分发
```cpp
// 根据 tensor->backend 查找对应的内核
const OpKernel* kernel = registry->get(backend, op_type);
kernel->execute(tensor);
```

### ✅ 3. 类型安全
```cpp
// 编译时检查：CUDA 算子只在 GENLLM_HAS_CUDA 时编译
#if GENLLM_HAS_CUDA
    REGISTER_BACKEND_KERNELS(Backend::CUDA, get_cuda_kernels());
#endif
```

### ✅ 4. 易于扩展
```cpp
// 添加新后端只需：
// 1. 实现 xxx_ops.h
// 2. 注册算子
// 3. 在 tensor->backend 中标记
```

## 内存池关联

```
Tensor* tensor
    ↓
tensor->backend → 指向 IMemoryResource (CpuMemoryResource/CudaMemoryResource)
    ↓
tensor->data → 由 MemoryPool 分配（从对应的 IMemoryResource）
    ↓
dispatch(tensor) → 根据 backend 选择内核
    ↓
CPU kernel → 操作 CPU 内存 (tensor->data)
CUDA kernel → 操作 GPU 内存 (tensor->data)
```

## 优势总结

| 特性 | 说明 |
|------|------|
| **自动分发** | 无需手动判断 backend |
| **编译时安全** | 不存在的后端无法编译 |
| **运行时高效** | 查找表 O(1) |
| **易于扩展** | 新增后端只需注册 |
| **解耦合** | 算子实现与调度器分离 |

## 对比其他方案

### ❌ 方案1：if-else 判断
```cpp
if (backend == CPU) {
    cpu::relu_forward(...);
} else if (backend == CUDA) {
    cuda::relu_forward(...);
}
```
**问题**：每个算子都要写一遍，不扩展

### ❌ 方案2：虚函数
```cpp
class DeviceContext {
    virtual void relu(Tensor*) = 0;
};
```
**问题**：需要为每个算子定义虚函数

### ✅ 方案3：注册表（我们采用的）
```cpp
registry.register(CPU, RELU, cpu_relu);
registry.register(CUDA, RELU, cuda_relu);
registry.get(backend, op)->execute();
```
**优势**：灵活、高效、易扩展
