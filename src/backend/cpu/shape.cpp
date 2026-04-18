#include <cstring>
#include <format>
#include <stdexcept>
#include "backend/cpu/shape.h"
#include "tensor.hpp"

namespace cpu {

void reshape(Tensor* out) {
    const Tensor* x = out->src[0];

    // 共享内存（零拷贝视图）
    out->data   = x->data;
    out->offset = x->offset;

    // 按新的 dims 重算行优先步长
    size_t stride = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (out->dims[i] == 0) {
            out->strides[i] = 0;
        } else {
            out->strides[i] = stride * data_type_size(out->dtype);
            stride *= static_cast<size_t>(out->dims[i]);
        }
    }
}

void permute(Tensor* out) {
    const Tensor* x = out->src[0];
    // 有效维度数
    int ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) {
        ndim = i + 1;
    }
    // 反推置换: out->dims[i] = x->dims[perm[i]]
    int perm[TENSOR_MAX_DIMS];
    bool used[TENSOR_MAX_DIMS]{};
    for (int i = 0; i < ndim; ++i) {
        perm[i] = -1;
        for (int j = 0; j < ndim; ++j) {
            if (!used[j] && out->dims[i] == x->dims[j]) {
                perm[i] = j;
                used[j] = true;
                break;
            }
        }
    }
    size_t elem_sz  = data_type_size(out->dtype);
    size_t total    = out->num_elements();
    auto* dst = static_cast<uint8_t*>(out->data);
    auto* src = static_cast<const uint8_t*>(x->data);
    size_t inner_dim   = static_cast<size_t>(out->dims[ndim - 1]);
    size_t inner_bytes = inner_dim * elem_sz;
    if (perm[ndim - 1] == ndim - 1) {
        // ── 快速路径: 最内维未被置换，源和输出都是行内连续 ──
        // 按外层维度遍历，每次拷贝一整行（inner_dim 个元素）
        size_t outer_count = total / inner_dim;
        for (size_t r = 0; r < outer_count; ++r) {
            // 将外层线性索引分解为多维坐标（不含最内维）
            size_t remaining  = r;
            size_t src_offset = 0;
            for (int d = ndim - 2; d >= 0; --d) {
                size_t dim_sz = static_cast<size_t>(out->dims[d]);
                size_t coord  = remaining % dim_sz;
                remaining /= dim_sz;
                src_offset += coord * x->strides[perm[d]];
            }
            std::memcpy(dst + r * inner_bytes, src + src_offset, inner_bytes);
        }
    } else {
        // ── 慢速路径: 最内维被置换，退化为逐元素拷贝 ──
        for (size_t idx = 0; idx < total; ++idx) {
            size_t remaining  = idx;
            size_t src_offset = 0;
            for (int d = ndim - 1; d >= 0; --d) {
                size_t dim_sz = static_cast<size_t>(out->dims[d]);
                size_t coord  = remaining % dim_sz;
                remaining /= dim_sz;
                src_offset += coord * x->strides[perm[d]];
            }
            std::memcpy(dst + idx * elem_sz, src + src_offset, elem_sz);
        }
    }
}

void concat(Tensor* out) {
    // TODO: concatenate tensors along axis
    // src[0], src[1]: tensors to concat
    // op_params[0]: axis
    throw std::runtime_error(std::format("cpu::concat not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void repeat(Tensor* out) {
    // TODO: repeat tensor along dimension
    // src[0]: input
    // op_params[0]: repeats count, op_params[1]: axis
    throw std::runtime_error(std::format("cpu::repeat not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cpu
