#include "context.hpp"
#include <algorithm>
#include "hostmemory.h"

// ==================== TensorContext 实现 ====================

template<typename T>
TensorContext<T>::TensorContext() : m_alignment(DEFAULT_TENSOR_ALIGNMENT) {}

template<typename T>
TensorContext<T>::TensorContext(size_t mem_size, size_t alignment) : m_alignment(alignment) {
    if (alignment < DEFAULT_TENSOR_ALIGNMENT) {
        throw std::invalid_argument("alignment must be at least 8 bytes");
    }
    m_mem_pool = T(mem_size, alignment);
}

template<typename T>
TensorContext<T>::~TensorContext() {
    // 回收 Tensor 对象
    for (auto t : m_tensors) {
        delete t;
    }
}

template<typename T>
size_t TensorContext<T>::alignment() const noexcept {
    return m_alignment;
}

template<typename T>
size_t TensorContext<T>::used_memory() const noexcept {
    return m_mem_pool.used();
}

template<typename T>
size_t TensorContext<T>::total_memory() const noexcept {
    return m_mem_pool.capacity();
}

template<typename T>
double TensorContext<T>::usage_ratio() const noexcept {
    return total_memory() > 0 ? static_cast<double>(used_memory()) / total_memory() : 0.0;
}

template<typename T>
void TensorContext<T>::reset() noexcept {
    m_mem_pool.reset();
}

template<typename T>
Tensor* TensorContext<T>::new_tensor(DataType dtype, std::initializer_list<int64_t> dims) {
    if (dims.size() > TENSOR_MAX_DIMS) {
        throw std::invalid_argument(std::format("Tensor dimensions exceed maximum of {}", TENSOR_MAX_DIMS));
    }
    auto* t = new Tensor();
    t->dtype = dtype;
    t->type = TensorType::OBJECT_TYPE_TENSOR;
    t->op_type = OperationType::OP_TYPE_NONE;
    std::copy(dims.begin(), dims.end(), t->dims.begin());
    // 计算 strides（默认紧凑存储）
    size_t stride = 1;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        t->strides[i] = stride * data_type_size(dtype);
        stride *= dims.begin()[i];
    }
    // 分配内存
    size_t total_size = stride * data_type_size(dtype);
    t->data = m_mem_pool.allocate(total_size, m_alignment);
    t->offset = reinterpret_cast<std::byte*>(t->data) - reinterpret_cast<std::byte*>(m_mem_pool.base_ptr());
    if (!t->data) {
        delete t;
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
    m_tensors.push_back(t);
    return t;
}

template<typename T>
Tensor* TensorContext<T>::new_tensor_1d(DataType dtype, int64_t ne0) {
    return new_tensor(dtype, {ne0});
}

template<typename T>
Tensor* TensorContext<T>::new_tensor_2d(DataType dtype, int64_t ne0, int64_t ne1) {
    return new_tensor(dtype, {ne0, ne1});
}

template<typename T>
Tensor* TensorContext<T>::new_tensor_3d(DataType dtype, int64_t ne0, int64_t ne1, int64_t ne2) {
    return new_tensor(dtype, {ne0, ne1, ne2});
}

template<typename T>
Tensor* TensorContext<T>::new_tensor_4d(DataType dtype, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return new_tensor(dtype, {ne0, ne1, ne2, ne3});
}

template<typename T>
void TensorContext<T>::allocate_tensor_memory(Tensor* t) {
    // 计算 tensor 总字节数
    size_t total_elements = 1;
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
        if (t->dims[i] == 0) break;  // 维度为 0 表示不使用
        total_elements *= t->dims[i];
    }
    size_t total_size = total_elements * data_type_size(t->dtype);

    // 从内存池分配
    t->data = m_mem_pool.allocate(total_size, m_alignment);
    t->offset = reinterpret_cast<std::byte*>(t->data) - reinterpret_cast<std::byte*>(m_mem_pool.base_ptr());

    if (!t->data) {
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
}

template<typename T>
Tensor* TensorContext<T>::create_op_node(Tensor* a, OperationType op_type) {
    auto* t = new Tensor();
    t->dtype = a->dtype;
    t->type = TensorType::OBJECT_TYPE_TENSOR;
    t->op_type = op_type;
    t->dims = a->dims;
    // 重新计算 strides
    size_t stride = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (t->dims[i] == 0) {
            t->strides[i] = 0;
        } else {
            t->strides[i] = stride * data_type_size(t->dtype);
            stride *= t->dims[i];
        }
    }
    t->src[0] = a;
    // 分配内存
    allocate_tensor_memory(t);
    m_tensors.push_back(t);
    return t;
}

template<typename T>
Tensor* TensorContext<T>::create_binary_op_node(Tensor* a, Tensor* b, OperationType op_type) {
    auto* t = new Tensor();
    t->dtype = a->dtype;
    t->type = TensorType::OBJECT_TYPE_TENSOR;
    t->op_type = op_type;
    // 广播规则：取较大的形状
    for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
        t->dims[i] = std::max(a->dims[i], b->dims[i]);
    }
    // 重新计算 strides
    size_t stride = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (t->dims[i] == 0) {
            t->strides[i] = 0;
        } else {
            t->strides[i] = stride * data_type_size(t->dtype);
            stride *= t->dims[i];
        }
    }
    t->src[0] = a;
    t->src[1] = b;
    // 分配内存
    allocate_tensor_memory(t);
    m_tensors.push_back(t);
    return t;
}

template<typename T>
void TensorContext<T>::set_param(Tensor* tensor) noexcept {
    if (tensor) {
        tensor->type = TensorType::OBJECT_TYPE_TENSOR;  // 可以扩展为 PARAM 类型
    }
}

template<typename T>
Tensor* TensorContext<T>::add(Tensor* a, Tensor* b) {
    return create_binary_op_node(a, b, OperationType::OP_TYPE_ADD);
}

template<typename T>
Tensor* TensorContext<T>::sub(Tensor* a, Tensor* b) {
    return create_binary_op_node(a, b, OperationType::OP_TYPE_SUB);
}

template<typename T>
Tensor* TensorContext<T>::mul(Tensor* a, Tensor* b) {
    return create_binary_op_node(a, b, OperationType::OP_TYPE_MUL);
}

template<typename T>
Tensor* TensorContext<T>::div(Tensor* a, Tensor* b) {
    return create_binary_op_node(a, b, OperationType::OP_TYPE_DIV);
}

template<typename T>
Tensor* TensorContext<T>::scale(Tensor* a, float scale) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_SCALE);
    union { float f; int64_t i; } u = {.f = scale};
    t->op_params[0] = u.i;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::mul_mat(Tensor* a, Tensor* b) {
    auto* t = new Tensor();
    t->dtype = a->dtype;
    t->type = TensorType::OBJECT_TYPE_TENSOR;
    t->op_type = OperationType::OP_TYPE_MUL_MAT;
    // 矩阵乘法输出: (M, K) * (K, N) -> (M, N)
    t->dims[0] = a->dims[0];  // M
    t->dims[1] = b->dims[1];  // N
    t->dims[2] = 0;
    t->dims[3] = 0;
    // 计算 strides
    t->strides[0] = t->dims[1] * data_type_size(t->dtype);
    t->strides[1] = data_type_size(t->dtype);
    t->strides[2] = 0;
    t->strides[3] = 0;
    t->src[0] = a;
    t->src[1] = b;
    // 分配内存
    allocate_tensor_memory(t);
    m_tensors.push_back(t);
    return t;
}

template<typename T>
Tensor* TensorContext<T>::transpose(Tensor* a) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_TRANSPOSE);
    // 交换前两个维度
    if (a->dims[1] > 0) {
        t->dims[0] = a->dims[1];
        t->dims[1] = a->dims[0];
    }
    return t;
}

template<typename T>
Tensor* TensorContext<T>::reshape(Tensor* a, std::initializer_list<int64_t> new_shape) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_RESHAPE);
    std::copy(new_shape.begin(), new_shape.end(), t->dims.begin());
    return t;
}

template<typename T>
Tensor* TensorContext<T>::view(Tensor* a, std::initializer_list<int64_t> new_shape) {
    auto* t = new Tensor();
    t->dtype = a->dtype;
    t->type = TensorType::OBJECT_TYPE_TENSOR;
    t->op_type = OperationType::OP_TYPE_VIEW;
    std::copy(new_shape.begin(), new_shape.end(), t->dims.begin());
    // 重新计算 strides
    size_t stride = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (t->dims[i] == 0) {
            t->strides[i] = 0;
        } else {
            t->strides[i] = stride * data_type_size(t->dtype);
            stride *= t->dims[i];
        }
    }
    t->src[0] = a;
    // 共享内存，不分配新空间
    t->data = a->data;
    t->offset = a->offset;
    m_tensors.push_back(t);
    return t;
}

template<typename T>
Tensor* TensorContext<T>::cont(Tensor* a) {
    return create_op_node(a, OperationType::OP_TYPE_DUP);
}

template<typename T>
Tensor* TensorContext<T>::concat(std::initializer_list<Tensor*> tensors, int dim) {
    if (dim >= TENSOR_MAX_DIMS) {
        throw std::invalid_argument("concat dimension out of range");
    }
    auto* t = new Tensor();
    t->dtype = tensors.begin()[0]->dtype;
    t->type = TensorType::OBJECT_TYPE_TENSOR;
    t->op_type = OperationType::OP_TYPE_CONCAT;
    // 复制第一个 tensor 的形状
    auto* first = tensors.begin()[0];
    t->dims = first->dims;
    // 计算拼接后的维度大小
    int64_t total_dim = 0;
    for (auto* tensor : tensors) {
        total_dim += tensor->dims[dim];
    }
    t->dims[dim] = total_dim;
    t->op_params[0] = dim;  // 保存拼接维度
    // 重新计算 strides
    size_t stride = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (t->dims[i] == 0) {
            t->strides[i] = 0;
        } else {
            t->strides[i] = stride * data_type_size(t->dtype);
            stride *= t->dims[i];
        }
    }
    // 设置源 tensors
    size_t idx = 0;
    for (auto* tensor : tensors) {
        if (idx < TENSOR_MAX_SRC) {
            t->src[idx++] = tensor;
        }
    }
    // 分配内存
    allocate_tensor_memory(t);
    m_tensors.push_back(t);
    return t;
}

template<typename T>
Tensor* TensorContext<T>::repeat(Tensor* a, std::initializer_list<int64_t> repeats) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_REPEAT);
    size_t idx = 0;
    for (auto r : repeats) {
        if (idx < TENSOR_MAX_DIMS) {
            t->dims[idx] = a->dims[idx] * r;
            t->op_params[idx] = r;
            ++idx;
        }
    }
    return t;
}

template<typename T>
Tensor* TensorContext<T>::softmax(Tensor* a, int dim) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_SOFTMAX);
    t->op_params[0] = dim;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::rms_norm(Tensor* a, float eps) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_RMS_NORM);
    union { float f; int64_t i; } u = {.f = eps};
    t->op_params[0] = u.i;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::layer_norm(Tensor* a, float eps) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_LAYER_NORM);
    union { float f; int64_t i; } u = {.f = eps};
    t->op_params[0] = u.i;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::gelu(Tensor* a) {
    return create_op_node(a, OperationType::OP_TYPE_GELU);
}

template<typename T>
Tensor* TensorContext<T>::silu(Tensor* a) {
    return create_op_node(a, OperationType::OP_TYPE_SILU);
}

template<typename T>
Tensor* TensorContext<T>::relu(Tensor* a) {
    return create_op_node(a, OperationType::OP_TYPE_RELU);
}

template<typename T>
Tensor* TensorContext<T>::get_rows(Tensor* a, Tensor* row_indices) {
    auto* t = new Tensor();
    t->dtype = a->dtype;
    t->type = TensorType::OBJECT_TYPE_TENSOR;
    t->op_type = OperationType::OP_TYPE_GET_ROWS;
    // 输出形状: (row_indices->dims[0], a->dims[1])
    t->dims[0] = row_indices->dims[0];
    t->dims[1] = a->dims[1];
    t->dims[2] = 0;
    t->dims[3] = 0;
    // 计算 strides
    t->strides[0] = t->dims[1] * data_type_size(t->dtype);
    t->strides[1] = data_type_size(t->dtype);
    t->strides[2] = 0;
    t->strides[3] = 0;
    t->src[0] = a;
    t->src[1] = row_indices;
    // 分配内存
    allocate_tensor_memory(t);
    m_tensors.push_back(t);
    return t;
}

template<typename T>
Tensor* TensorContext<T>::diag_mask_inf(Tensor* a, int n_past) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_DIAG_MASK_INF);
    t->op_params[0] = n_past;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::pool_2d(Tensor* a, int k0, int k1, int s0, int s1,
                                 enum OperationType pool_type) {
    auto* t = create_op_node(a, pool_type);
    // 计算输出形状: (H - k0) / s0 + 1, (W - k1) / s1 + 1
    if (a->dims[0] > 0 && a->dims[1] > 0) {
        t->dims[0] = (a->dims[0] - k0) / s0 + 1;
        t->dims[1] = (a->dims[1] - k1) / s1 + 1;
    }
    t->op_params[0] = k0;
    t->op_params[1] = k1;
    t->op_params[2] = s0;
    t->op_params[3] = s1;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::upscale(Tensor* a, int scale_factor) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_UPSCALE);
    if (a->dims[0] > 0 && a->dims[1] > 0) {
        t->dims[0] = a->dims[0] * scale_factor;
        t->dims[1] = a->dims[1] * scale_factor;
    }
    t->op_params[0] = scale_factor;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::pad(Tensor* a, int pad_left, int pad_right, int pad_top, int pad_bottom) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_PAD);
    if (a->dims[0] > 0 && a->dims[1] > 0) {
        t->dims[0] = a->dims[0] + pad_top + pad_bottom;
        t->dims[1] = a->dims[1] + pad_left + pad_right;
    }
    t->op_params[0] = pad_left;
    t->op_params[1] = pad_right;
    t->op_params[2] = pad_top;
    t->op_params[3] = pad_bottom;
    return t;
}

template<typename T>
Tensor* TensorContext<T>::unpad(Tensor* a, int left, int right, int top, int bottom) {
    auto* t = create_op_node(a, OperationType::OP_TYPE_UNPAD);
    if (a->dims[0] > 0 && a->dims[1] > 0) {
        t->dims[0] = a->dims[0] - top - bottom;
        t->dims[1] = a->dims[1] - left - right;
    }
    t->op_params[0] = left;
    t->op_params[1] = right;
    t->op_params[2] = top;
    t->op_params[3] = bottom;
    return t;
}

// 显式实例化 - 为 HostMemory 类型实例化模板
template class TensorContext<HostMemory>;
