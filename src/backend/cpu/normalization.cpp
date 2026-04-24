#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include "utils/bfloat16.hpp"
#include "utils/dtype_traits.hpp"
#include "utils/float16.hpp"
#include "backend/cpu/normalization.h"



// 假设 bfloat16_t 的类定义已经包含（如之前的头文件）
// 这里直接使用 bfloat16_t 类型

namespace {
    // 加载 8 个 bfloat16 → 8 个 float
    inline __m256 load_bf16_f32(const bfloat16_t* ptr) {
        const uint16_t* raw = reinterpret_cast<const uint16_t*>(ptr);
        __m128i bf16 = _mm_loadu_si128((const __m128i*)raw);
        __m256i i32  = _mm256_cvtepu16_epi32(bf16);
        i32 = _mm256_slli_epi32(i32, 16);
        return _mm256_castsi256_ps(i32);
    }

    // 将 8 个 float 存储为 bfloat16（截断低16位）
    inline void store_f32_bf16(bfloat16_t* ptr, __m256 v) {
        __m256i shifted = _mm256_srli_epi32(_mm256_castps_si256(v), 16);
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        __m128i packed = _mm_packus_epi32(lo, hi);   // 顺序正确 [0..7]
        _mm_storeu_si128((__m128i*)ptr, packed);
    }

    // 8 个 float 水平求和
    inline float hsum_ps(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
} // anonymous namespace

void rms_norm(
    bfloat16_t* out,
    const bfloat16_t* x,
    const float* w,
    size_t seq_len,
    size_t hidden_size,
    float eps)
{
    constexpr size_t VEC_WIDTH = 8;        // AVX2 一次处理 8 个 float
    constexpr size_t UNROLL = 4;          // 平方和累加器展开
    for (size_t i = 0; i < seq_len; ++i) {
        const bfloat16_t* x_row = x + i * hidden_size;
        bfloat16_t* out_row = out + i * hidden_size;
        // ----- 1. 计算平方和 -----
        size_t j = 0;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        // 主循环：每次处理 32 个元素 (4×8)
        for (; j + UNROLL * VEC_WIDTH <= hidden_size; j += UNROLL * VEC_WIDTH) {
            __m256 x0 = load_bf16_f32(x_row + j);
            __m256 x1 = load_bf16_f32(x_row + j + 8);
            __m256 x2 = load_bf16_f32(x_row + j + 16);
            __m256 x3 = load_bf16_f32(x_row + j + 24);
            acc0 = _mm256_fmadd_ps(x0, x0, acc0);
            acc1 = _mm256_fmadd_ps(x1, x1, acc1);
            acc2 = _mm256_fmadd_ps(x2, x2, acc2);
            acc3 = _mm256_fmadd_ps(x3, x3, acc3);
        }
        // 合并累加器
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);
        // 剩余 8 的倍数
        for (; j + VEC_WIDTH <= hidden_size; j += VEC_WIDTH) {
            __m256 xv = load_bf16_f32(x_row + j);
            acc0 = _mm256_fmadd_ps(xv, xv, acc0);
        }
        float sum_sq = hsum_ps(acc0);
        // 标量尾巴
        for (; j < hidden_size; ++j) {
            float val = float(x_row[j]);
            sum_sq += val * val;
        }
        // ----- 2. 计算 RMS 缩放因子 -----
        float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        __m256 rms_vec = _mm256_set1_ps(rms);

        // ----- 3. 缩放并写回 -----
        j = 0;
        for (; j + VEC_WIDTH <= hidden_size; j += VEC_WIDTH) {
            __m256 xv = load_bf16_f32(x_row + j);
            __m256 wv = _mm256_loadu_ps(w + j);
            __m256 result = _mm256_mul_ps(_mm256_mul_ps(xv, rms_vec), wv);
            store_f32_bf16(out_row + j, result);
        }
        // 标量尾巴
        for (; j < hidden_size; ++j) {
            float val = float(x_row[j]) * rms * w[j];
            out_row[j] = bfloat16_t(val);   // 利用 bfloat16_t(float) 构造器完成 RNE 舍入
        }
    }
}
// out = (x - mean) / std * w + b
// T: x/out 的数据类型，w 和 b 为 float
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void layer_norm(T* out, const T* x, const float* w, const float* b, size_t num_tokens, size_t hidden_size, float eps) {
    for (size_t t = 0; t < num_tokens; ++t) {
        const T* x_row = x + t * hidden_size;
        T* o_row = out + t * hidden_size;

        float mean = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i)
            mean += static_cast<float>(x_row[i]);
        mean /= static_cast<float>(hidden_size);

        float var = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float d = static_cast<float>(x_row[i]) - mean;
            var += d * d;
        }
        float inv_std = 1.0f / std::sqrt(var / static_cast<float>(hidden_size) + eps);

        for (size_t i = 0; i < hidden_size; ++i) {
            float fx = static_cast<float>(x_row[i]);
            float fb = b ? b[i] : 0.0f;
            o_row[i] = static_cast<T>((fx - mean) * inv_std * w[i] + fb);
        }
    }
}

namespace ops {

void RmsNormImpl<Device::CPU>::execute(Tensor* out) {
    const Tensor* x = out->src[0];  // [batch, seq_len, hidden_size] or  [batch, num_heads, seq_len, head_dim]
    const Tensor* w = out->src[1];  // [hidden_size]                 or  [head_dim]
    float eps = out->op_params[0];
    size_t hidden_size = w->num_elements();
    size_t total_tokens = x->num_elements() / hidden_size;   // 总行数
    rms_norm(
        reinterpret_cast<bfloat16_t*>(out->data),
        reinterpret_cast<const bfloat16_t*>(x->data),
        reinterpret_cast<const float*>(w->data),
        total_tokens,
        hidden_size,
        eps
    );

}

    void LayerNormImpl<Device::CPU>::execute(Tensor* out) {
        const Tensor* x = out->src[0];
        const Tensor* w = out->src[1];
        const Tensor* b = out->src[2];
        float eps = out->op_params[0];
        size_t hidden_size = w->num_elements();
        size_t num_tokens = x->num_elements() / hidden_size;
        const float* bp = b && b->data ? static_cast<const float*>(b->data) : nullptr;
        dtype::dispatch(x->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            layer_norm(static_cast<T*>(out->data), static_cast<const T*>(x->data),
                       static_cast<const float*>(w->data), bp, num_tokens, hidden_size, eps);
        });
    }

template struct RmsNormImpl<Device::CPU>;
template struct LayerNormImpl<Device::CPU>;
}
