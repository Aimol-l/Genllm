#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <stdexcept>
#include "backend/cpu/linear.h"
#include "utils/bfloat16.hpp"
#include "utils/dtype_traits.hpp"
#include "utils/float16.hpp"

// ---------- 辅助工具函数 ----------
// 将两个连续的 128-bit (8×fp16) 转换为两个 256-bit (8×fp32)
inline void load_fp16_x2(__m256& lo, __m256& hi, const float16_t* addr) {
    __m128i half_lo = _mm_loadu_si128((const __m128i*)(addr));
    __m128i half_hi = _mm_loadu_si128((const __m128i*)(addr + 8));
    lo = _mm256_cvtph_ps(half_lo);
    hi = _mm256_cvtph_ps(half_hi);
}
// 单次加载 8 个 fp16 转为 8 个 fp32
inline __m256 load_fp16(const float16_t* addr) {
    __m128i half = _mm_loadu_si128((const __m128i*)addr);
    return _mm256_cvtph_ps(half);
}
// 8×fp32 水平求和
inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
// bfloat16_t 内部只有一个 uint16_t 成员，内存布局与 uint16_t 兼容
static inline __m256 load_bf16x8_f32(const bfloat16_t* ptr) {
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(ptr);
    __m128i bf16 = _mm_loadu_si128((const __m128i*)raw);
    __m256i i32  = _mm256_cvtepu16_epi32(bf16);   // 零扩展至 32bit
    i32 = _mm256_slli_epi32(i32, 16);              // 移至 float 的高 16 位
    return _mm256_castsi256_ps(i32);
}
// 加载 16 个 bf16 → 两个 8×float 向量（双累加器）
static inline void load_bf16_x2(__m256 &lo, __m256 &hi, const bfloat16_t* ptr) {
    lo = load_bf16x8_f32(ptr);
    hi = load_bf16x8_f32(ptr + 8);
}

template <typename T>
void linear(
    bfloat16_t* __restrict__ out,
    const bfloat16_t* __restrict__ x,    // [rows, common], row-major
    const bfloat16_t* __restrict__ w,    // [cols, common], row-major
    int64_t rows,
    int64_t common,
    int64_t cols,
    const bfloat16_t* __restrict__ bias  // [cols]
) {
    constexpr int64_t MR = 16;    // 输出行分块
    constexpr int64_t NR = 16;    // 输出列分块（8 个 float = 256-bit）
    constexpr int64_t KR = 256;  // K 维度缓存分块
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < rows; i0 += MR) {
        int64_t i1 = std::min(i0 + MR, rows);
        int64_t mr = i1 - i0;
        for (int64_t j0 = 0; j0 < cols; j0 += NR) {
            int64_t j1 = std::min(j0 + NR, cols);
            int64_t nr = j1 - j0;
            // FP32 累加器 [MR][NR]，初始 0
            alignas(32) float accum[MR][NR] = {0.0f};
            // K 分块循环
            for (int64_t k0 = 0; k0 < common; k0 += KR) {
                int64_t k1 = std::min(k0 + KR, common);
                int64_t k_len = k1 - k0;
                // 当前 K 块内，逐行逐列做向量化点积
                for (int64_t r = 0; r < mr; ++r) {
                    const bfloat16_t* x_ptr = x + (i0 + r) * common + k0;
                    for (int64_t c = 0; c < nr; ++c) {
                        const bfloat16_t* w_ptr = w + (j0 + c) * common + k0;
                        float local_sum = accum[r][c];
                        int64_t k = 0;
                        // 双累加器：一次处理 16 个 bf16
                        if (k_len >= 16) {
                            __m256 sum0 = _mm256_setzero_ps();
                            __m256 sum1 = _mm256_setzero_ps();
                            for (; k + 16 <= k_len; k += 16) {
                                __m256 x0, x1, w0, w1;
                                load_bf16_x2(x0, x1, x_ptr + k);
                                load_bf16_x2(w0, w1, w_ptr + k);
                                sum0 = _mm256_fmadd_ps(x0, w0, sum0);
                                sum1 = _mm256_fmadd_ps(x1, w1, sum1);
                            }
                            sum0 = _mm256_add_ps(sum0, sum1);
                            local_sum += hsum_ps(sum0);
                        }
                        // 单累加器：处理剩余 8 的倍数
                        for (; k + 8 <= k_len; k += 8) {
                            __m256 x_vec = load_bf16x8_f32(x_ptr + k);
                            __m256 w_vec = load_bf16x8_f32(w_ptr + k);
                            __m256 sum   = _mm256_fmadd_ps(x_vec, w_vec, _mm256_setzero_ps());
                            local_sum += hsum_ps(sum);
                        }
                        // 标量尾部（<8 元素）
                        for (; k < k_len; ++k) {
                            // 利用你重载的 float() 进行隐式转换
                            local_sum += float(x_ptr[k]) * float(w_ptr[k]);
                        }
                        accum[r][c] = local_sum;
                    }
                }
            }
            // 添加 bias 并写回 bfloat16
            for (int64_t r = 0; r < mr; ++r) {
                for (int64_t c = 0; c < nr; ++c) {
                    float val = accum[r][c];
                    if (bias) {
                        val += float(bias[j0 + c]);   // 列广播
                    }
                    // 使用你提供的 RNE 舍入构造 bfloat16_t
                    out[(i0 + r) * cols + (j0 + c)] = bfloat16_t(val);
                }
            }
        }
    }
}

template <typename T>
void linear(
    float16_t* __restrict__ out,
    const float16_t* __restrict__ x,    // [rows, common], row-major
    const float16_t* __restrict__ w,    // [cols, common], row-major
    int64_t rows,
    int64_t common,
    int64_t cols,
    const float16_t* __restrict__ bias  // [cols]
) {
    constexpr int64_t MR = 8;    // 输出行分块
    constexpr int64_t NR = 8;    // 输出列分块（对应 8×fp32 寄存器宽度）
    constexpr int64_t KR = 256;  // K 维度分块（L2 友好）
    // 外层：遍历输出行块
    for (int64_t i0 = 0; i0 < rows; i0 += MR) {
        int64_t i1 = std::min(i0 + MR, rows);
        int64_t mr = i1 - i0;   // 实际行数 (1..8)
        // 中层：遍历输出列块
        for (int64_t j0 = 0; j0 < cols; j0 += NR) {
            int64_t j1 = std::min(j0 + NR, cols);
            int64_t nr = j1 - j0;   // 实际列数 (1..8)
            // FP32 累加器，形状 [MR][NR]，初始为 0
            alignas(32) float accum[MR][NR] = {0.0f}; // 一块的大小
            // 计算出accum这一小块的结果
            for (int64_t k0 = 0; k0 < common; k0 += KR) {
                int64_t k1 = std::min(k0 + KR, common);
                int64_t k_len = k1 - k0;
                // 对当前 K 块，对每一个输出行和列进行向量化点积
                for (int64_t r = 0; r < mr; ++r) {
                    const float16_t* x_ptr = x + (i0 + r) * common + k0;
                    for (int64_t c = 0; c < nr; ++c) {
                        const float16_t* w_ptr = w + (j0 + c) * common + k0;
                        float local_sum = accum[r][c];
                        int64_t k = 0;
                        // 双累加器：同时处理 16 个 fp16（转换后得到 16 个 fp32）
                        if (k_len >= 16) {
                            __m256 sum0 = _mm256_setzero_ps();
                            __m256 sum1 = _mm256_setzero_ps();
                            for (; k + 16 <= k_len; k += 16) {
                                __m256 x0, x1, w0, w1;
                                load_fp16_x2(x0, x1, x_ptr + k);
                                load_fp16_x2(w0, w1, w_ptr + k);
                                sum0 = _mm256_fmadd_ps(x0, w0, sum0);
                                sum1 = _mm256_fmadd_ps(x1, w1, sum1);
                            }
                            sum0 = _mm256_add_ps(sum0, sum1);
                            local_sum += hsum_ps(sum0);
                        }
                        // 处理剩余 8 元素块（或 k_len < 16）
                        for (; k + 8 <= k_len; k += 8) {
                            __m256 x_vec = load_fp16(x_ptr + k);
                            __m256 w_vec = load_fp16(w_ptr + k);
                            // 单累加器 FMA
                            __m256 sum = _mm256_setzero_ps();
                            sum = _mm256_fmadd_ps(x_vec, w_vec, sum);
                            local_sum += hsum_ps(sum);
                        }
                        // 尾部标量处理（<8 元素）
                        for (; k < k_len; ++k) {
                            local_sum += float(x_ptr[k]) * float(w_ptr[k]);
                        }
                        accum[r][c] = local_sum;
                    }
                }
            }
            // 小块算完后，直接复制到结果数组里面
            for (int64_t r = 0; r < mr; ++r) {
                for (int64_t c = 0; c < nr; ++c) {
                    float val = accum[r][c];
                    if (bias) {
                        val += float(bias[j0 + c]);  // bias 按列广播
                    }
                    out[(i0 + r) * cols + (j0 + c)] = float16_t(val);
                }
            }
        }
    }
}
template <typename T>
void linear(
    float* __restrict__ out,
    const float* __restrict__ x,    // [rows, common], row-major
    const float* __restrict__ w,    // [cols, common], row-major
    int64_t rows,
    int64_t common,
    int64_t cols,
    const float* __restrict__ bias  // [cols]
) {
    constexpr int64_t MR = 8;    // 输出行分块
    constexpr int64_t NR = 8;    // 输出列分块（8 个 float 填满 256-bit 寄存器）
    constexpr int64_t KR = 256;  // K 维度缓存分块
    #pragma omp parallel for schedule(static)
    for (int64_t i0 = 0; i0 < rows; i0 += MR) {
        int64_t i1 = std::min(i0 + MR, rows);
        int64_t mr = i1 - i0;
        for (int64_t j0 = 0; j0 < cols; j0 += NR) {
            int64_t j1 = std::min(j0 + NR, cols);
            int64_t nr = j1 - j0;
            // FP32 累加器 [MR][NR]，初始 0
            alignas(32) float accum[MR][NR] = {0.0f};
            // K 分块循环
            for (int64_t k0 = 0; k0 < common; k0 += KR) {
                int64_t k1 = std::min(k0 + KR, common);
                int64_t k_len = k1 - k0;
                // 对当前 K 块，逐行逐列做向量化点积
                for (int64_t r = 0; r < mr; ++r) {
                    const float* x_ptr = x + (i0 + r) * common + k0;
                    for (int64_t c = 0; c < nr; ++c) {
                        const float* w_ptr = w + (j0 + c) * common + k0;
                        float local_sum = accum[r][c];
                        int64_t k = 0;
                        // 双累加器：一次处理 16 个 float (2 × 8)
                        if (k_len >= 16) {
                            __m256 sum0 = _mm256_setzero_ps();
                            __m256 sum1 = _mm256_setzero_ps();
                            for (; k + 16 <= k_len; k += 16) {
                                __m256 x0 = _mm256_loadu_ps(x_ptr + k);
                                __m256 x1 = _mm256_loadu_ps(x_ptr + k + 8);
                                __m256 w0 = _mm256_loadu_ps(w_ptr + k);
                                __m256 w1 = _mm256_loadu_ps(w_ptr + k + 8);
                                sum0 = _mm256_fmadd_ps(x0, w0, sum0);
                                sum1 = _mm256_fmadd_ps(x1, w1, sum1);
                            }
                            sum0 = _mm256_add_ps(sum0, sum1);
                            local_sum += hsum_ps(sum0);
                        }
                        // 单累加器：处理剩余 8 的倍数
                        for (; k + 8 <= k_len; k += 8) {
                            __m256 x_vec = _mm256_loadu_ps(x_ptr + k);
                            __m256 w_vec = _mm256_loadu_ps(w_ptr + k);
                            __m256 sum   = _mm256_fmadd_ps(x_vec, w_vec, _mm256_setzero_ps());
                            local_sum += hsum_ps(sum);
                        }
                        // 标量尾部（<8 元素）
                        for (; k < k_len; ++k) {
                            local_sum += x_ptr[k] * w_ptr[k];
                        }
                        accum[r][c] = local_sum;
                    }
                }
            }
            // 添加 bias 并写回 float
            for (int64_t r = 0; r < mr; ++r) {
                for (int64_t c = 0; c < nr; ++c) {
                    float val = accum[r][c];
                    if (bias) {
                        val += bias[j0 + c];   // 列广播
                    }
                    out[(i0 + r) * cols + (j0 + c)] = val;
                }
            }
        }
    }
}


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
        const Tensor* bias = out->src[2]; // ...
        bool transpose_w = out->op_params[0] == 1;

        int64_t batch = x->dims[0];

        assert(batch == 1);

        dtype::dispatch(w->dtype, [&]<DataType D_w>() {
            using Tw = dtype::type_t<D_w>;
            auto* op = static_cast<Tw*>(out->data);
            auto* xp = static_cast<const Tw*>(x->data);
            auto* wp = static_cast<const Tw*>(w->data);
            auto* bp = bias && bias->data ? static_cast<const Tw*>(bias->data) : nullptr;
            if(transpose_w){ // 尽管在数学上需要先将w进行转置，但是我们在具体实践中不要。
                int64_t rows    = x->dims[1];
                int64_t common  = w->dims[1];
                int64_t cols    = w->dims[0];
                linear<Tw>(op, xp, wp, rows, common, cols, bp);
            }else{
                throw std::runtime_error("Linear transpose = false not impl");
            }
        });
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