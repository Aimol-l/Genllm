#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include "backend/cpu/linear.h"
#include "utils/bfloat16.hpp"
#include "utils/dtype_traits.hpp"
#include "utils/float16.hpp"
#include "utils/quants.hpp"
#include "utils/tensor_dump.hpp"


// ---------- 辅助工具函数 ----------
// 批量转换 8 个 float16 -> 8 个 float32
static inline __m256 load8_halfs(const float16_t* ptr) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return _mm256_cvtph_ps(h);   // 一条指令完成 8 个 fp16->fp32
}
// 将 8 个 float32 存储为 8 个 float16（如果需要写回）
static inline void store8_halfs(float16_t* ptr, __m256 f32) {
    __m128i h = _mm256_cvtps_ph(f32, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), h);
}
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
static inline uint8_t get_6bit(const std::byte* scales, int i) {
    int byte_off = (i * 6) / 8;
    int bit_off  = (i * 6) % 8;
    uint16_t val = (static_cast<uint16_t>(scales[byte_off]) << 8) |
                   static_cast<uint16_t>(scales[byte_off + 1]);
    return (val >> (10 - bit_off)) & 0x3F;   // 大端/小端需根据实际布局调整
}
// 安全加载 8 个 fp16 -> 8 个 float（不对齐）
inline __m256 load_fp16_unaligned(const float16_t* ptr) {
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    return _mm256_cvtph_ps(h);
}

// 安全加载 16 个 fp16 -> 两个 256bit 向量
inline void load_fp16_x2_unaligned(__m256& a, __m256& b, const float16_t* ptr) {
    __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    a = _mm256_cvtph_ps(_mm256_extracti128_si256(h, 0));
    b = _mm256_cvtph_ps(_mm256_extracti128_si256(h, 1));
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
    const float16_t* __restrict__ x,    // [M, K]
    const float16_t* __restrict__ w,    // [N, K]
    int64_t M, int64_t K, int64_t N,
    const float16_t* __restrict__ bias  // [N], can be nullptr
) {
    constexpr int64_t BM = 8;
    constexpr int64_t BN = 8;
    constexpr int64_t BK = 256;
    for (int64_t i0 = 0; i0 < M; i0 += BM) {
        int64_t mb = std::min(BM, M - i0);
        for (int64_t j0 = 0; j0 < N; j0 += BN) {
            int64_t nb = std::min(BN, N - j0);
            // 用 float32 累加，防止量化或大矩阵带来的精度问题
            float accum[BM][BN] = {{0.0f}};
            // 沿 K 方向分块累加
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                int64_t kb = std::min(BK, K - k0);
                // 标量微内核
                for (int64_t r = 0; r < mb; ++r) {
                    const float16_t* x_row = x + (i0 + r) * K + k0;
                    for (int64_t c = 0; c < nb; ++c) {
                        const float16_t* w_row = w + (j0 + c) * K + k0;
                        float dot = accum[r][c];
                        for (int64_t k = 0; k < kb; ++k) {
                            dot += float(x_row[k]) * float(w_row[k]);
                        }
                        accum[r][c] = dot;
                    }
                }
            }
            // 写回（带 bias 和防溢出）
            for (int64_t r = 0; r < mb; ++r) {
                for (int64_t c = 0; c < nb; ++c) {
                    float val = accum[r][c];
                    if (bias) {
                        val += float(bias[j0 + c]);
                    }
                    out[(i0 + r) * N + (j0 + c)] = float16_t(val);
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

template <typename T>
void linear(
    bfloat16_t* __restrict__ out,
    const bfloat16_t* __restrict__ x,          // [M, K]
    const Block_Q4_K* __restrict__ w,          // [N * (K/256)]
    int64_t M, int64_t K, int64_t N,
    const bfloat16_t* __restrict__ bias        // [N], 可选
) {
    constexpr int64_t BM = 8;
    constexpr int64_t BN = 8;
    constexpr int64_t BK = 256;
    const int super_blocks = K / BK;

    // bias 初始化（顺序执行，O(M*N) 相比 O(M*N*K) 可忽略）
    for (int64_t i = 0; i < M; ++i) {
        if (bias) std::copy_n(bias, N, out + i * N);
        else      std::fill_n(out + i * N, N, bfloat16_t(0.0f));
    }

    // 微内核：计算行块 [ii, ii+mb) × 列块 [jj, jj+nb) 的累加贡献
    auto compute_block = [&](int64_t ii, int64_t mb, int64_t jj, int64_t nb) {
        float temp[BM][BN] = {{0.0f}};
        for (int sb = 0; sb < super_blocks; ++sb) {
            alignas(32) bfloat16_t w_buf[BN][BK];
            for (int64_t c = 0; c < nb; ++c)
                quant::dequantize_row_q4_K(w + (jj + c) * super_blocks + sb, w_buf[c], BK);
            for (int64_t r = 0; r < mb; ++r) {
                const bfloat16_t* x_ptr = x + (ii + r) * K + sb * BK;
                for (int64_t c = 0; c < nb; ++c) {
                    const bfloat16_t* w_ptr = w_buf[c];
                    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                    for (int64_t k = 0; k < BK; k += 16) {
                        __m256 xv0 = load_bf16x8_f32(x_ptr + k);
                        __m256 xv1 = load_bf16x8_f32(x_ptr + k + 8);
                        __m256 wv0 = load_bf16x8_f32(w_ptr + k);
                        __m256 wv1 = load_bf16x8_f32(w_ptr + k + 8);
                        s0 = _mm256_fmadd_ps(xv0, wv0, s0);
                        s1 = _mm256_fmadd_ps(xv1, wv1, s1);
                    }
                    temp[r][c] += hsum_ps(_mm256_add_ps(s0, s1));
                }
            }
        }
        for (int64_t r = 0; r < mb; ++r)
            for (int64_t c = 0; c < nb; ++c) {
                float old = float(out[(ii + r) * N + (jj + c)]);
                out[(ii + r) * N + (jj + c)] = bfloat16_t(old + temp[r][c]);
            }
    };

    if (M >= BM * 2) {
        // Prefill：行数足够，并行行块
        #pragma omp parallel for schedule(static)
        for (int64_t ii = 0; ii < M; ii += BM) {
            const int64_t mb = std::min(BM, M - ii);
            for (int64_t jj = 0; jj < N; jj += BN)
                compute_block(ii, mb, jj, std::min(BN, N - jj));
        }
    } else {
        // Decode（M=1 等）：行太少，并行列块
        for (int64_t ii = 0; ii < M; ii += BM) {
            const int64_t mb = std::min(BM, M - ii);
            #pragma omp parallel for schedule(static)
            for (int64_t jj = 0; jj < N; jj += BN)
                compute_block(ii, mb, jj, std::min(BN, N - jj));
        }
    }
}

template <typename T>
void linear(
    bfloat16_t* __restrict__ out,
    const bfloat16_t* __restrict__ x,          // [M, K]
    const Block_Q6_K* __restrict__ w,          // [N * (K/256)]
    int64_t M, int64_t K, int64_t N,
    const bfloat16_t* __restrict__ bias        // [N], optional
) {
    constexpr int64_t BM = 8;
    constexpr int64_t BN = 8;
    constexpr int64_t BK = 256;
    const int super_blocks = K / BK;

    for (int64_t i = 0; i < M; ++i) {
        if (bias) std::copy_n(bias, N, out + i * N);
        else      std::fill_n(out + i * N, N, bfloat16_t(0.0f));
    }

    auto compute_block = [&](int64_t ii, int64_t mb, int64_t jj, int64_t nb) {
        float temp[BM][BN] = {{0.0f}};
        for (int sb = 0; sb < super_blocks; ++sb) {
            alignas(32) bfloat16_t w_buf[BN][BK];
            for (int64_t c = 0; c < nb; ++c)
                quant::dequantize_row_q6_K(w + (jj + c) * super_blocks + sb, w_buf[c], BK);
            for (int64_t r = 0; r < mb; ++r) {
                const bfloat16_t* x_ptr = x + (ii + r) * K + sb * BK;
                for (int64_t c = 0; c < nb; ++c) {
                    const bfloat16_t* w_ptr = w_buf[c];
                    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                    for (int64_t k = 0; k < BK; k += 16) {
                        __m256 xv0 = load_bf16x8_f32(x_ptr + k);
                        __m256 xv1 = load_bf16x8_f32(x_ptr + k + 8);
                        __m256 wv0 = load_bf16x8_f32(w_ptr + k);
                        __m256 wv1 = load_bf16x8_f32(w_ptr + k + 8);
                        s0 = _mm256_fmadd_ps(xv0, wv0, s0);
                        s1 = _mm256_fmadd_ps(xv1, wv1, s1);
                    }
                    temp[r][c] += hsum_ps(_mm256_add_ps(s0, s1));
                }
            }
        }
        for (int64_t r = 0; r < mb; ++r)
            for (int64_t c = 0; c < nb; ++c) {
                float old = float(out[(ii + r) * N + (jj + c)]);
                out[(ii + r) * N + (jj + c)] = bfloat16_t(old + temp[r][c]);
            }
    };

    if (M >= BM * 2) {
        #pragma omp parallel for schedule(static)
        for (int64_t ii = 0; ii < M; ii += BM) {
            const int64_t mb = std::min(BM, M - ii);
            for (int64_t jj = 0; jj < N; jj += BN)
                compute_block(ii, mb, jj, std::min(BN, N - jj));
        }
    } else {
        for (int64_t ii = 0; ii < M; ii += BM) {
            const int64_t mb = std::min(BM, M - ii);
            #pragma omp parallel for schedule(static)
            for (int64_t jj = 0; jj < N; jj += BN)
                compute_block(ii, mb, jj, std::min(BN, N - jj));
        }
    }
}

template <typename T>
void linear(
    bfloat16_t* __restrict__ out,
    const bfloat16_t* __restrict__ x,          // [M, K]
    const Block_IQ4_NL* __restrict__ w,        // [N * (K/32)]
    int64_t M, int64_t K, int64_t N,
    const bfloat16_t* __restrict__ bias        // [N], optional
) {
    constexpr int64_t BM = 8;
    constexpr int64_t BN = 8;
    constexpr int64_t BK = QK4_NL;
    const int64_t blocks_per_row = K / BK;

    for (int64_t i = 0; i < M; ++i) {
        if (bias) std::copy_n(bias, N, out + i * N);
        else      std::fill_n(out + i * N, N, bfloat16_t(0.0f));
    }

    auto compute_block = [&](int64_t ii, int64_t mb, int64_t jj, int64_t nb) {
        float temp[BM][BN] = {{0.0f}};
        for (int64_t sb = 0; sb < blocks_per_row; ++sb) {
            alignas(32) bfloat16_t w_buf[BN][BK];
            for (int64_t c = 0; c < nb; ++c)
                quant::dequantize_row_iq4_nl(w + (jj + c) * blocks_per_row + sb, w_buf[c], BK);
            for (int64_t r = 0; r < mb; ++r) {
                const bfloat16_t* x_ptr = x + (ii + r) * K + sb * BK;
                for (int64_t c = 0; c < nb; ++c) {
                    const bfloat16_t* w_ptr = w_buf[c];
                    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                    for (int64_t k = 0; k < BK; k += 16) {
                        __m256 xv0 = load_bf16x8_f32(x_ptr + k);
                        __m256 xv1 = load_bf16x8_f32(x_ptr + k + 8);
                        __m256 wv0 = load_bf16x8_f32(w_ptr + k);
                        __m256 wv1 = load_bf16x8_f32(w_ptr + k + 8);
                        s0 = _mm256_fmadd_ps(xv0, wv0, s0);
                        s1 = _mm256_fmadd_ps(xv1, wv1, s1);
                    }
                    temp[r][c] += hsum_ps(_mm256_add_ps(s0, s1));
                }
            }
        }
        for (int64_t r = 0; r < mb; ++r)
            for (int64_t c = 0; c < nb; ++c) {
                float old = float(out[(ii + r) * N + (jj + c)]);
                out[(ii + r) * N + (jj + c)] = bfloat16_t(old + temp[r][c]);
            }
    };

    if (M >= BM * 2) {
        #pragma omp parallel for schedule(static)
        for (int64_t ii = 0; ii < M; ii += BM) {
            const int64_t mb = std::min(BM, M - ii);
            for (int64_t jj = 0; jj < N; jj += BN)
                compute_block(ii, mb, jj, std::min(BN, N - jj));
        }
    } else {
        for (int64_t ii = 0; ii < M; ii += BM) {
            const int64_t mb = std::min(BM, M - ii);
            #pragma omp parallel for schedule(static)
            for (int64_t jj = 0; jj < N; jj += BN)
                compute_block(ii, mb, jj, std::min(BN, N - jj));
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

namespace ops {

    void LinearImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x    = out->src[0]; // bf16，f16,..
        const Tensor* w    = out->src[1]; // bf16,量化
        const Tensor* bias = out->src[2]; // ...

        int64_t M    = x->dims[1]; 
        int64_t K    = x->dims[2]; 
        int64_t N    = w->dims[0];

        auto* bias_ptr = bias && bias->data ? static_cast<const bfloat16_t*>(bias->data) : nullptr;

        switch(w->dtype){
            case DataType::GGML_TYPE_F32:
                linear<float>(static_cast<float*>(out->data),static_cast<const float*>(x->data),static_cast<const float*>(w->data), M, K, N, nullptr);break;
            case DataType::GGML_TYPE_F16: 
                linear<float16_t>(static_cast<float16_t*>(out->data),static_cast<const float16_t*>(x->data),static_cast<const float16_t*>(w->data), M, K, N, nullptr);break;
            case DataType::GGML_TYPE_BF16:
                linear<bfloat16_t>(static_cast<bfloat16_t*>(out->data),static_cast<const bfloat16_t*>(x->data),static_cast<const bfloat16_t*>(w->data),M, K, N, nullptr);break;
            case DataType::GGML_TYPE_Q4_K:{
                linear<Block_Q4_K>(static_cast<bfloat16_t*>(out->data),static_cast<const bfloat16_t*>(x->data),static_cast<const Block_Q4_K*>(w->data), M, K, N, bias_ptr);break;
            }
            case DataType::GGML_TYPE_Q6_K:{
                linear<Block_Q6_K>(static_cast<bfloat16_t*>(out->data),static_cast<const bfloat16_t*>(x->data),static_cast<const Block_Q6_K*>(w->data), M, K, N, bias_ptr);break;
            }
            case DataType::GGML_TYPE_IQ4_NL:{
                linear<Block_IQ4_NL>(static_cast<bfloat16_t*>(out->data),static_cast<const bfloat16_t*>(x->data),static_cast<const Block_IQ4_NL*>(w->data), M, K, N, bias_ptr);break;
            }
            default: 
                throw std::runtime_error(std::format("linear: unsupported dtype {}", data_type_to_string(w->dtype)));
        }
        // tensor_dump::dump_tensor_npy(x,"dump/linear_x.npy");
        // tensor_dump::dump_tensor_npy(w,"dump/linear_weight.npy");
    }

    void MatmulImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
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

template struct LinearImpl<Device::CPU>;
template struct MatmulImpl<Device::CPU>;
}
