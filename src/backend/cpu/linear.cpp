#include <cstring>
#include <format>
#include <stdexcept>
#include <vector>
#include "backend/cpu/linear.h"
#include "utils/dtype_traits.hpp"

namespace cpu {

static bool is_floating(DataType dt) {
    return dt == DataType::GGML_TYPE_F32 ||
           dt == DataType::GGML_TYPE_F16 ||
           dt == DataType::GGML_TYPE_BF16;
}

// ────────────────────────────────────────────────────────────────
//  matmul: C = A @ B
//  src[0]: A [M, K], src[1]: B [K, N]
// ────────────────────────────────────────────────────────────────
void matmul(Tensor* out) {
    const Tensor* a = out->src[0];
    const Tensor* b = out->src[1];

    if (!is_floating(a->dtype) || !is_floating(b->dtype)) {
        throw std::runtime_error(std::format(
            "cpu::matmul: quantized dtype not supported yet (a={}, b={})",
            data_type_to_string(a->dtype), data_type_to_string(b->dtype)));
    }

    int64_t M = a->dims[0];
    int64_t K = a->dims[1];
    int64_t N = b->dims[1];
    int64_t b_ld = b->dims[1];

    dtype::dispatch(b->dtype, [&]<DataType D_b>() {
        using Tb = dtype::type_t<D_b>;
        const Tb* bp = static_cast<const Tb*>(b->data);

        size_t a_sz = data_type_size(a->dtype);
        size_t o_sz = data_type_size(out->dtype);
        auto* ap = static_cast<const uint8_t*>(a->data);
        auto* op = static_cast<uint8_t*>(out->data);

        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += dtype::to_f32_rt(a->dtype, ap + (m * K + k) * a_sz)
                         * dtype::to_f32<D_b>(bp[k * b_ld + n]);
                }
                dtype::from_f32_rt(out->dtype, sum,
                    op + (m * N + n) * o_sz);
            }
        }
    });
}

// ────────────────────────────────────────────────────────────────
//  linear: y = x @ W + bias
//  src[0]: x [..., in_features]
//  src[1]: W [in_features, out_features]
//  src[2]: bias [out_features] (可选)
// ────────────────────────────────────────────────────────────────
void linear(Tensor* out) {
    const Tensor* x    = out->src[0];
    const Tensor* w    = out->src[1];
    const Tensor* bias = out->src[2];

    bool transpose_w = out->op_params[0] == 1;

    if (!is_floating(w->dtype)) {
        throw std::runtime_error(std::format(
            "cpu::linear: quantized weight ({}) not supported yet",
            data_type_to_string(w->dtype)));
    }

    int64_t in_features, out_features, w_ld;
    if (transpose_w) {
        // W 存储为 [out_features, in_features]
        out_features = w->dims[0];
        in_features  = w->dims[1];
    } else {
        // W 存储为 [in_features, out_features]
        in_features  = w->dims[0];
        out_features = w->dims[1];
    }
    w_ld = w->dims[1]; // 列宽（每行元素数）
    size_t M = x->num_elements() / static_cast<size_t>(in_features);

    size_t x_sz = data_type_size(x->dtype);
    size_t o_sz = data_type_size(out->dtype);
    auto* xp = static_cast<const uint8_t*>(x->data);
    auto* op = static_cast<uint8_t*>(out->data);

    const uint8_t* bp    = nullptr;
    size_t         b_sz  = 0;
    if (bias && bias->data) {
        bp   = static_cast<const uint8_t*>(bias->data);
        b_sz = data_type_size(bias->dtype);
    }

    dtype::dispatch(w->dtype, [&]<DataType D_w>() {
        using Tw = dtype::type_t<D_w>;
        const Tw* wp = static_cast<const Tw*>(w->data);

        std::vector<float> x_row(static_cast<size_t>(in_features));

        for (size_t m = 0; m < M; ++m) {
            // 1. 当前行 input 转为 float（每行只做一次）
            for (int64_t k = 0; k < in_features; ++k) {
                x_row[static_cast<size_t>(k)] = dtype::to_f32_rt(
                    x->dtype, xp + (m * in_features + k) * x_sz);
            }
            // 2. x_row @ W[:, j] + bias
            for (int64_t j = 0; j < out_features; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < in_features; ++k) {
                    float w_val = transpose_w
                        ? dtype::to_f32<D_w>(wp[j * w_ld + k])
                        : dtype::to_f32<D_w>(wp[k * w_ld + j]);
                    sum += x_row[static_cast<size_t>(k)] * w_val;
                }
                if (bp) {
                    sum += dtype::to_f32_rt(bias->dtype, bp + j * b_sz);
                }
                dtype::from_f32_rt(out->dtype, sum,
                    op + (m * out_features + j) * o_sz);
            }
        }
    });
}

// ────────────────────────────────────────────────────────────────
//  transpose: 交换两个轴
//  src[0]: input
//  op_params[0], op_params[1]: 要交换的两个轴
// ────────────────────────────────────────────────────────────────
void transpose(Tensor* out) {
    const Tensor* x = out->src[0];
    int64_t ax0 = static_cast<int64_t>(out->op_params[0]);
    int64_t ax1 = static_cast<int64_t>(out->op_params[1]);

    int ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) {
        ndim = i + 1;
    }

    int64_t src_dims[TENSOR_MAX_DIMS]{};
    uint64_t src_strides[TENSOR_MAX_DIMS]{};
    for (int i = 0; i < ndim; ++i) {
        src_dims[i]    = x->dims[i];
        src_strides[i] = x->strides[i];
    }
    std::swap(src_dims[ax0], src_dims[ax1]);
    std::swap(src_strides[ax0], src_strides[ax1]);

    size_t elem_sz = data_type_size(x->dtype);
    size_t total   = out->num_elements();

    for (size_t idx = 0; idx < total; ++idx) {
        size_t remaining  = idx;
        size_t src_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            size_t dim_size = static_cast<size_t>(src_dims[d]);
            size_t coord    = remaining % dim_size;
            remaining /= dim_size;
            src_offset += coord * src_strides[d];
        }
        std::memcpy(
            static_cast<uint8_t*>(out->data) + idx * elem_sz,
            static_cast<const uint8_t*>(x->data) + src_offset,
            elem_sz);
    }
}

} // namespace cpu
