// operator.h - 算子分发和内核注册系统（无宏版本）
#pragma once
#include <unordered_map>
#include <functional>
#include <memory>
#include <print>
#include <format>
#include "tensor.hpp"
#include "utils/utils.hpp"
#include "backend.h"

// ==================== 算子内核接口 ====================

// 算子执行上下文
struct OpContext {
    const Device* device;       // 目标设备
    void* stream;               // 执行流（CUDA stream等）
    void* workspace;            // 临时工作空间
    size_t workspace_size;      // 工作空间大小

    OpContext() : device(nullptr), stream(nullptr), workspace(nullptr), workspace_size(0) {}

    OpContext(const Device* dev)
        : device(dev), stream(nullptr), workspace(nullptr), workspace_size(0) {}
};

// 算子内核函数签名
using OpKernelFunction = std::function<void(Tensor*, const OpContext&)>;

// 算子内核描述
struct OpKernel {
    Backend backend;              // 所属后端
    OperationType op_type;        // 操作类型
    std::string name;             // 内核名称
    OpKernelFunction func;        // 执行函数
    bool is_available;            // 是否可用

    OpKernel() : backend(Backend::CPU), op_type(OperationType::OP_TYPE_NONE),
                 is_available(false) {}

    OpKernel(Backend be, OperationType op, const std::string& n, OpKernelFunction f)
        : backend(be), op_type(op), name(n), func(f), is_available(true) {}

    // 执行算子
    void execute(Tensor* tensor, const OpContext& ctx) const {
        if (!is_available) {
            throw std::runtime_error(std::format(
                "Kernel {}::{} is not available",
                static_cast<int>(backend),
                operation_type_to_string(op_type)));
        }

        if (!func) {
            throw std::runtime_error(std::format(
                "Kernel function is null for {}::{}",
                static_cast<int>(backend),
                operation_type_to_string(op_type)));
        }

        func(tensor, ctx);
    }

    [[nodiscard]] std::string to_string() const {
        return std::format("{}::{}", static_cast<int>(backend),
                          operation_type_to_string(op_type));
    }
};

// ==================== 算子注册表 ====================

class OpRegistry {
private:
    // 键: (backend, op_type) -> 值: OpKernel
    std::unordered_map<uint64_t, OpKernel> m_kernels;

    // 生成键
    static uint64_t make_key(Backend backend, OperationType op_type) {
        return (static_cast<uint64_t>(backend) << 32) | static_cast<uint64_t>(op_type);
    }

public:
    OpRegistry() = default;
    ~OpRegistry() = default;

    // 禁止拷贝
    OpRegistry(const OpRegistry&) = delete;
    OpRegistry& operator=(const OpRegistry&) = delete;

    // 注册单个算子内核
    void register_kernel(const OpKernel& kernel) {
        uint64_t key = make_key(kernel.backend, kernel.op_type);
        m_kernels[key] = kernel;
        std::println("Registered kernel: {}", kernel.to_string());
    }

    // 注册算子内核（使用显式参数）
    void register_kernel(Backend backend, OperationType op_type,
                        const std::string& name, OpKernelFunction func) {
        register_kernel(OpKernel(backend, op_type, name, func));
    }

    // 批量注册某个后端的所有内核
    void register_kernels(Backend backend, const std::vector<OpKernel>& kernels) {
        for (const auto& kernel : kernels) {
            if (kernel.backend == backend) {
                register_kernel(kernel);
            }
        }
    }

    // ==================== 内核查找 ====================

    // 获取算子内核
    [[nodiscard]] const OpKernel* get_kernel(Backend backend,
                                             OperationType op_type) const {
        uint64_t key = make_key(backend, op_type);
        auto it = m_kernels.find(key);
        return it != m_kernels.end() ? &it->second : nullptr;
    }

    // 检查内核是否可用
    [[nodiscard]] bool has_kernel(Backend backend, OperationType op_type) const {
        return get_kernel(backend, op_type) != nullptr;
    }

    // ==================== 调试信息 ====================

    // 打印所有已注册的内核
    void print_kernels() const {
        std::println("\n=== Registered Operator Kernels ({}) ===", m_kernels.size());

        // 按后端分组
        std::unordered_map<Backend, std::vector<const OpKernel*>> backend_groups;
        for (const auto& [key, kernel] : m_kernels) {
            backend_groups[kernel.backend].push_back(&kernel);
        }

        for (const auto& [backend, kernels] : backend_groups) {
            std::println("\nBackend {}:", static_cast<int>(backend));
            for (const auto* kernel : kernels) {
                std::println("  - {}", operation_type_to_string(kernel->op_type));
            }
        }
    }

    // 获取统计信息
    [[nodiscard]] size_t kernel_count() const {
        return m_kernels.size();
    }

    [[nodiscard]] size_t backend_count(Backend backend) const {
        size_t count = 0;
        for (const auto& [key, kernel] : m_kernels) {
            if (kernel.backend == backend) {
                count++;
            }
        }
        return count;
    }
};

// ==================== 全局算子注册表 ====================

inline OpRegistry& get_op_registry() {
    static OpRegistry registry;
    return registry;
}

// ==================== 算子分发器 ====================

class OpDispatcher {
private:
    const OpRegistry* m_registry;

public:
    OpDispatcher() : m_registry(&get_op_registry()) {}

    // 执行算子（自动选择对应后端的内核）
    void dispatch(Tensor* tensor, const OpContext& ctx = OpContext()) {
        if (!tensor) {
            throw std::runtime_error("Cannot dispatch null tensor");
        }

        // 从 tensor->backend 获取后端类型
        Backend backend = get_tensor_backend(tensor);

        // 查找对应的内核
        const OpKernel* kernel = m_registry->get_kernel(backend, tensor->op_type);

        if (!kernel) {
            throw std::runtime_error(std::format(
                "No kernel found for {}::{}",
                static_cast<int>(backend),
                operation_type_to_string(tensor->op_type)));
        }

        // 执行内核
        std::println("Dispatching: {} on {}",
            operation_type_to_string(tensor->op_type),
            static_cast<int>(backend));

        kernel->execute(tensor, ctx);
    }

    // 执行算子（显式指定后端）
    void dispatch_on(Tensor* tensor, Backend backend, const OpContext& ctx = OpContext()) {
        if (!tensor) {
            throw std::runtime_error("Cannot dispatch null tensor");
        }

        const OpKernel* kernel = m_registry->get_kernel(backend, tensor->op_type);

        if (!kernel) {
            throw std::runtime_error(std::format(
                "No kernel found for {}::{}",
                static_cast<int>(backend),
                operation_type_to_string(tensor->op_type)));
        }

        kernel->execute(tensor, ctx);
    }

private:
    // 从张量获取后端类型
    static Backend get_tensor_backend(Tensor* tensor) {
        if (!tensor || !tensor->backend) {
            return Backend::CPU;  // 默认 CPU
        }
        // 从 IMemoryResource 获取 Backend
        IMemoryResource* resource = static_cast<IMemoryResource*>(tensor->backend);
        return resource ? resource->get_backend() : Backend::CPU;
    }
};

// 全局分发器
inline OpDispatcher& get_op_dispatcher() {
    static OpDispatcher dispatcher;
    return dispatcher;
}

inline void register_op(OpKernelFunction func,OperationType optype,std::string name,Backend backend) {
    get_op_registry().register_kernel(backend,optype,name,func);
}