#include <cstdint>
#include <immintrin.h>
#include "backend/cpu/linear.h"
#include "utils/bfloat16.hpp"
#include "utils/dtype_traits.hpp"
#include "utils/float16.hpp"


// ============ AVX2 Kernel for float (8-wide FMA) ============
inline void linear_float_avx2(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ w,
    int64_t rows,
    int64_t common,
    int64_t cols,
    const float* __restrict__ bias
) {
    constexpr int64_t BM = 4;   // M维度块大小
    constexpr int64_t BN = 8;   // N维度块大小 (AVX2: 8 floats)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t m0 = 0; m0 < rows; m0 += BM) {
        for (int64_t n0 = 0; n0 < cols; n0 += BN) {
            // 寄存器累加器 [BM][BN/8]
            __m256 acc[BM];
            for (int64_t i = 0; i < BM; ++i) acc[i] = _mm256_setzero_ps();
            // K维度主循环
            for (int64_t k = 0; k < common; ++k) {
                // 预取下一轮w数据（可选，视数据大小启用）
                // _mm_prefetch(&w[(k+1)*cols + n0], _MM_HINT_T0);
                for (int64_t i = 0; i < BM; ++i) {
                    int64_t m = m0 + i;
                    if (m >= rows) continue;
                    // 广播 x[m,k]
                    __m256 x_val = _mm256_broadcast_ss(&x[m * common + k]);
                    // 加载 w[k, n0:n0+BN]
                    if (n0 + BN <= cols) {
                        __m256 w_vec = _mm256_loadu_ps(&w[k * cols + n0]);
                        acc[i] = _mm256_fmadd_ps(x_val, w_vec, acc[i]);
                    }
                }
            }
            // 写回结果 + bias
            for (int64_t i = 0; i < BM; ++i) {
                int64_t m = m0 + i;
                if (m >= rows) continue;
                int64_t n_limit = std::min(n0 + BN, cols);
                float tmp[8];
                _mm256_storeu_ps(tmp, acc[i]);  // 提取标量
                for (int64_t n = n0; n < n_limit; ++n) {
                    float sum = tmp[n - n0];
                    if (bias) sum += bias[n];
                    out[m * cols + n] = sum;
                }
            }
        }
    }
}

// ============ 通用入口（支持 float/float16/bfloat16） ============
template <typename T> 
requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t> || std::is_same_v<T, float>
void linear(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ w,
    int64_t rows,
    int64_t common,
    int64_t cols,
    const T* __restrict__ bias
) {
    if constexpr (std::is_same_v<T, float>) {
        // 小矩阵或无法向量化时降级
        if (rows < 4 || cols < 8) {
            goto fallback;
        }
        linear_float_avx2(out, x, w, rows, common, cols, bias);
        return;
    }
    
fallback:
    // Half精度 / 小矩阵：分块 + OpenMP + float累加
    constexpr int64_t BM = 32, BN = 32;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t m0 = 0; m0 < rows; m0 += BM) {
        for (int64_t n0 = 0; n0 < cols; n0 += BN) {
            int64_t m_end = std::min(m0 + BM, rows);
            int64_t n_end = std::min(n0 + BN, cols);
            
            for (int64_t m = m0; m < m_end; ++m) {
                for (int64_t n = n0; n < n_end; ++n) {
                    float sum = 0.0f;
                    // K循环在最内层，配合编译器向量化
                    for (int64_t k = 0; k < common; ++k) {
                        sum += static_cast<float>(x[m * common + k]) * 
                               static_cast<float>(w[k * cols + n]);
                    }
                    if (bias) sum += static_cast<float>(bias[n]);
                    out[m * cols + n] = static_cast<T>(sum);
                }
            }
        }
    }
}
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void linear_T(
    T* out,
    const T* x,
    const T* w,
    int64_t rows,
    int64_t common,
    int64_t cols,
    const T* bias
){
    // out [rows, cols] = x [rows, common] @ w^T [cols, common] + bias
    for (int64_t m = 0; m < rows; ++m) {
        for (int64_t j = 0; j < cols; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < common; ++k) {
                sum += static_cast<float>(x[m * common + k]) * static_cast<float>(w[j * common + k]);
            }
            if (bias) sum += static_cast<float>(bias[j]);
            out[m * cols + j] = static_cast<T>(sum);
        }
    }
}

// ────────────────────────────────────────────────────────────────
//  matmul: out [M, N] = a [M, K] @ b [K, N]
// ────────────────────────────────────────────────────────────────
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void matmul(
    T* out,
    const T* a,
    const T* b,
    int64_t M,
    int64_t K,
    int64_t N
){
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<float>(a[m * K + k]) * static_cast<float>(b[k * N + n]);
            }
            out[m * N + n] = static_cast<T>(sum);
        }
    }
}
// ────────────────────────────────────────────────────────────────
//  transpose: 交换两个轴
// ────────────────────────────────────────────────────────────────
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void transpose(
    T* out,
    const T* in,
    const int64_t* src_dims,
    const uint64_t* src_strides,
    int ndim,
    size_t total
){
    for (size_t idx = 0; idx < total; ++idx) {
        size_t remaining  = idx;
        size_t src_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            size_t dim_size = static_cast<size_t>(src_dims[d]);
            size_t coord    = remaining % dim_size;
            remaining /= dim_size;
            src_offset += coord * src_strides[d];
        }
        out[idx] = in[src_offset];
    }
}

namespace ops {

    // [batch,rows,common] @ [common,cols] = [batch,rows,cols]
    // [batch,rows,common] @ [cols,common].T = [batch,rows,cols]
    void LinearImpl<Device::CPU>::execute(Tensor* out) {
        const Tensor* x    = out->src[0]; // bf16
        const Tensor* w    = out->src[1]; // bf16
        const Tensor* bias = out->src[2];
        bool transpose_w = out->op_params[0] == 1;

        dtype::dispatch(w->dtype, [&]<DataType D_w>() {
            using Tw = dtype::type_t<D_w>;
            auto* op = static_cast<Tw*>(out->data);
            auto* xp = static_cast<const Tw*>(x->data);
            auto* wp = static_cast<const Tw*>(w->data);
            auto* bp = bias && bias->data ? static_cast<const Tw*>(bias->data) : nullptr;
            if(transpose_w){
                // W [out_features, in_features]
                int64_t in_features  = w->dims[1];
                int64_t out_features = w->dims[0];
                int64_t rows = x->num_elements() / in_features;
                linear_T<Tw>(op, xp, wp, rows, in_features, out_features, bp);
            }else{
                // W [in_features, out_features]
                int64_t in_features  = w->dims[0];
                int64_t out_features = w->dims[1];
                int64_t rows = x->num_elements() / in_features;
                linear<Tw>(op, xp, wp, rows, in_features, out_features, bp);
            }
        });
        // const Tensor* x    = out->src[0];
        // const Tensor* w    = out->src[1];
        // const Tensor* bias = out->src[2];
        // bool transpose_w = out->op_params[0] == 1;
        // int64_t in_features, out_features, w_ld;
        // if (transpose_w) {
        //     // W 存储为 [out_features, in_features]
        //     out_features = w->dims[0];
        //     in_features  = w->dims[1];
        // } else {
        //     // W 存储为 [in_features, out_features]
        //     in_features  = w->dims[0];
        //     out_features = w->dims[1];
        // }
        // w_ld = w->dims[1]; // 列宽（每行元素数）
        // size_t M = x->num_elements() / static_cast<size_t>(in_features);
        // size_t x_sz = data_type_size(x->dtype);
        // size_t o_sz = data_type_size(out->dtype);
        // auto* xp = static_cast<const uint8_t*>(x->data);
        // auto* op = static_cast<uint8_t*>(out->data);
        // const uint8_t* bp    = nullptr;
        // size_t         b_sz  = 0;
        // if (bias && bias->data) {
        //     bp   = static_cast<const uint8_t*>(bias->data);
        //     b_sz = data_type_size(bias->dtype);
        // }
        // dtype::dispatch(w->dtype, [&]<DataType D_w>() {
        //     using Tw = dtype::type_t<D_w>;
        //     const Tw* wp = static_cast<const Tw*>(w->data);
        //     std::vector<float> x_row(static_cast<size_t>(in_features));
        //     for (size_t m = 0; m < M; ++m) {
        //         // 1. 当前行 input 转为 float（每行只做一次）
        //         for (int64_t k = 0; k < in_features; ++k) {
        //             x_row[static_cast<size_t>(k)] = dtype::to_f32_rt(
        //                 x->dtype, xp + (m * in_features + k) * x_sz);
        //         }
        //         // 2. x_row @ W[:, j] + bias
        //         for (int64_t j = 0; j < out_features; ++j) {
        //             float sum = 0.0f;
        //             for (int64_t k = 0; k < in_features; ++k) {
        //                 float w_val = transpose_w
        //                     ? dtype::to_f32<D_w>(wp[j * w_ld + k])
        //                     : dtype::to_f32<D_w>(wp[k * w_ld + j]);
        //                 sum += x_row[static_cast<size_t>(k)] * w_val;
        //             }
        //             if (bp) {
        //                 sum += dtype::to_f32_rt(bias->dtype, bp + j * b_sz);
        //             }
        //             dtype::from_f32_rt(out->dtype, sum,
        //                 op + (m * out_features + j) * o_sz);
        //         }
        //     }
        // });
    }

    void MatmulImpl<Device::CPU>::execute(Tensor* out) {
        const Tensor* a = out->src[0];
        const Tensor* b = out->src[1];
        dtype::dispatch(b->dtype, [&]<DataType D_b>() {
            using Tb = dtype::type_t<D_b>;
            auto* op = static_cast<Tb*>(out->data);
            auto* ap = static_cast<const Tb*>(a->data);
            auto* bp = static_cast<const Tb*>(b->data);
            matmul<Tb>(op, ap, bp, a->dims[0], a->dims[1], b->dims[1]);
        });
    }

    void TransposeImpl<Device::CPU>::execute(Tensor* out) {
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

        dtype::dispatch(x->dtype, [&]<DataType D_x>() {
            using Tx = dtype::type_t<D_x>;
            auto* op = static_cast<Tx*>(out->data);
            auto* xp = static_cast<const Tx*>(x->data);
            transpose<Tx>(op, xp, src_dims, src_strides, ndim, out->num_elements());
        });
    }

template struct LinearImpl<Device::CPU>;
template struct MatmulImpl<Device::CPU>;
template struct TransposeImpl<Device::CPU>;
}