#include <cassert>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <ranges>
#include <vector>
#include "backend/cpu/rope.h"
#include "utils/dtype_traits.hpp"

namespace cpu {

// 对 q/k 张量应用旋转位置编码 (RoPE),qwen3的实现方式，前半维与后半维配对旋转
void apply_rope(Tensor* out) {
    const Tensor* x   = out->src[0];  // [B, n_heads, seq_len, head_dim] , bf16/f16
    const Tensor* cos = out->src[1];  // [max_seq_len, head_dim] F32
    const Tensor* sin = out->src[2];  // [max_seq_len, head_dim] F32

    int64_t head_dim = static_cast<int64_t>(out->op_params[0]);
    int64_t half_dim = static_cast<int64_t>(out->op_params[0] / 2);
    int64_t start_pos = static_cast<int64_t>(out->op_params[2]);  // ✅ 新增

    int64_t B = x->dims[0], n_heads = x->dims[1], seq_len = x->dims[2];
    assert(B == 1);

    const float* cos_data = static_cast<const float*>(cos->data);
    const float* sin_data = static_cast<const float*>(sin->data);

    size_t cos_stride = head_dim;

    dtype::dispatch(out->dtype, [&]<DataType D_out>() {
        using T = dtype::type_t<D_out>;
        T* x_out = static_cast<T*>(out->data);
        const T* x_in = static_cast<const T*>(x->data);

        size_t head_stride  = static_cast<size_t>(seq_len * head_dim);
        size_t batch_stride = static_cast<size_t>(n_heads * seq_len * head_dim);

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < n_heads; ++h) {
                for (int64_t s = 0; s < seq_len; ++s) {

                    int64_t pos = start_pos + s;
                    const float* cos = cos_data + pos * cos_stride;
                    const float* sin = sin_data + pos * cos_stride;

                    T* row_out = x_out + b * batch_stride + h * head_stride + s * head_dim;
                    const T* row_in = x_in + b * batch_stride + h * head_stride + s * head_dim;

                    #pragma omp simd
                    for (int64_t i = 0; i < half_dim; ++i) {
                        float x0 = dtype::to_f32<D_out>(row_in[i]);
                        float x1 = dtype::to_f32<D_out>(row_in[i + half_dim]);
                        float ci = cos[i], si = sin[i];

                        float xr = x0 * ci - x1 * si;
                        float xi_rot = x0 * si + x1 * ci;

                        row_out[i] = dtype::from_f32<D_out>(xr);
                        row_out[i + half_dim] = dtype::from_f32<D_out>(xi_rot);
                    }
                }
            }
        }
    });
}

// void apply_rope(Tensor* out) {
//     const Tensor* x   = out->src[0];  // [B, n_heads, seq_len, head_dim]
//     const Tensor* cos = out->src[1];  // [max_seq_len, head_dim] 或 [max_seq_len, head_dim/2]
//     const Tensor* sin = out->src[2];  // 同上
//     const int32_t start_pos = static_cast<int32_t>(out->op_params[2]);  // KV Cache 起始绝对位置
//     // ========== 维度解析 ==========
//     const int B = x->dims[0];
//     const int n_heads = x->dims[1];
//     const int seq_len = x->dims[2];
//     const int head_dim = x->dims[3];
//     assert(B == 1 && "当前仅支持 batch=1，KV Cache 场景");
//     assert(head_dim % 2 == 0 && "head_dim 必须为偶数，RoPE 以 2 维为旋转单元");
    
    

// }
// out [max_seq_len, head_dim] 预计算 RoPE 的 cos/sin 表
void rope_cache(Tensor* out) {
    const float theta = out->op_params[0];
    const int head_dim = static_cast<int>(out->op_params[1]);
    const int max_seq = static_cast<int>(out->op_params[2]);
    const bool is_cos = out->name.find("_cos") != std::string::npos;
    
    const int half_dim = head_dim / 2;
    float* dst = static_cast<float*>(out->data);
    
    std::vector<double> inv_freqs(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        double exponent = (2.0 * static_cast<double>(i)) / static_cast<double>(head_dim);
        inv_freqs[i] = 1.0 / std::pow(static_cast<double>(theta), exponent);
    }
    
    for (int s = 0; s < max_seq; ++s) {
        const double pos = static_cast<double>(s);
        const int base_idx = s * head_dim;  // 输出 [max_seq, head_dim]
        for (int i = 0; i < half_dim; ++i) {
            const double angle = pos * inv_freqs[i];
            const float value = is_cos 
                ? static_cast<float>(std::cos(angle))
                : static_cast<float>(std::sin(angle));
            dst[base_idx + i] = value;
            dst[base_idx + i + half_dim] = value;
        }
    }
}
} // namespace cpu
