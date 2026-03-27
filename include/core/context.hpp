// context.hpp - Tensor 上下文和内存管理
#pragma once
#include <memory>
#include <vector>
#include <unordered_map>
#include <span>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <mutex>
#include "tensor.hpp"
#include "graph.hpp"
#include "memory.h"

constexpr size_t DEFAULT_TENSOR_ALIGNMENT = 8;

// Tensor 上下文模板类
template<typename T>
class TensorContext {
private:
    T m_mem_pool;
    std::vector<Tensor*> m_tensors;
    size_t m_alignment;
private:
    void allocate_tensor_memory(Tensor* t);
    [[nodiscard]] Tensor* create_op_node(Tensor* a, OperationType op_type);
    [[nodiscard]] Tensor* create_binary_op_node(Tensor* a, Tensor* b, OperationType op_type);
public:
    // 构造与析构
    TensorContext();
    explicit TensorContext(size_t mem_size, size_t alignment = DEFAULT_TENSOR_ALIGNMENT);
    ~TensorContext();
    // 禁止拷贝，允许移动
    TensorContext(const TensorContext&) = delete;
    TensorContext& operator=(const TensorContext&) = delete;
    TensorContext(TensorContext&&) noexcept = default;
    TensorContext& operator=(TensorContext&&) noexcept = default;
    // 状态查询
    [[nodiscard]] size_t alignment() const noexcept;
    [[nodiscard]] size_t used_memory() const noexcept;
    [[nodiscard]] size_t total_memory() const noexcept;
    [[nodiscard]] double usage_ratio() const noexcept;

    // 内存管理
    void reset() noexcept;
    // ==================== Tensor 创建（ggml 风格）====================
    [[nodiscard]] Tensor* new_tensor(DataType dtype, std::initializer_list<int64_t> dims);
    [[nodiscard]] Tensor* new_tensor_1d(DataType dtype, int64_t ne0);
    [[nodiscard]] Tensor* new_tensor_2d(DataType dtype, int64_t ne0, int64_t ne1);
    [[nodiscard]] Tensor* new_tensor_3d(DataType dtype, int64_t ne0, int64_t ne1, int64_t ne2);
    [[nodiscard]] Tensor* new_tensor_4d(DataType dtype, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
    // ==================== 计算图算子（ggml 风格）====================
    void set_param(Tensor* tensor) noexcept;
    [[nodiscard]] Tensor* add(Tensor* a, Tensor* b);
    [[nodiscard]] Tensor* sub(Tensor* a, Tensor* b);
    [[nodiscard]] Tensor* mul(Tensor* a, Tensor* b);
    [[nodiscard]] Tensor* div(Tensor* a, Tensor* b);
    [[nodiscard]] Tensor* scale(Tensor* a, float scale);
    [[nodiscard]] Tensor* mul_mat(Tensor* a, Tensor* b);
    [[nodiscard]] Tensor* transpose(Tensor* a);
    [[nodiscard]] Tensor* reshape(Tensor* a, std::initializer_list<int64_t> new_shape);
    [[nodiscard]] Tensor* view(Tensor* a, std::initializer_list<int64_t> new_shape);
    [[nodiscard]] Tensor* cont(Tensor* a);
    [[nodiscard]] Tensor* concat(std::initializer_list<Tensor*> tensors, int dim = 0);
    [[nodiscard]] Tensor* repeat(Tensor* a, std::initializer_list<int64_t> repeats);
    [[nodiscard]] Tensor* softmax(Tensor* a, int dim = -1);
    [[nodiscard]] Tensor* rms_norm(Tensor* a, float eps = 1e-5f);
    [[nodiscard]] Tensor* layer_norm(Tensor* a, float eps = 1e-5f);
    [[nodiscard]] Tensor* gelu(Tensor* a);
    [[nodiscard]] Tensor* silu(Tensor* a);
    [[nodiscard]] Tensor* relu(Tensor* a);
    [[nodiscard]] Tensor* get_rows(Tensor* a, Tensor* row_indices);
    [[nodiscard]] Tensor* diag_mask_inf(Tensor* a, int n_past);
    [[nodiscard]] Tensor* pool_2d(Tensor* a, int k0, int k1, int s0, int s1,enum OperationType pool_type = OperationType::OP_TYPE_POOL_2D);
    [[nodiscard]] Tensor* upscale(Tensor* a, int scale_factor);
    [[nodiscard]] Tensor* pad(Tensor* a, int pad_left, int pad_right, int pad_top, int pad_bottom);
    [[nodiscard]] Tensor* unpad(Tensor* a, int left, int right, int top, int bottom);
};
