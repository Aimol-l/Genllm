#include <cmath>
#include <format>
#include <stdexcept>
#include "backend/cpu/rope.h"
#include "utils/dtype_traits.hpp"

namespace cpu {

// 对 q/k 张量应用旋转位置编码 (RoPE)
void apply_rope(Tensor* out) {
    const Tensor* x   = out->src[0];  // [B, n_heads, seq_len, head_dim] BF16
    const Tensor* cos = out->src[1];  // [max_seq_len, half_dim]  F32
    const Tensor* sin = out->src[2];  // [max_seq_len, half_dim]  F32

    int64_t head_dim = static_cast<int64_t>(out->op_params[0]);
    int64_t half_dim = static_cast<int64_t>(out->op_params[1]);

    int64_t B   = x->dims[0];
    int64_t nh  = x->dims[1];
    int64_t S   = x->dims[2];
    int64_t D   = x->dims[3]; // == head_dim
    int64_t HD  = D / 2;      // == half_dim

    const float* cos_data = static_cast<const float*>(cos->data);
    const float* sin_data = static_cast<const float*>(sin->data);
    size_t cos_stride = HD; // cos/sin 缓存每行 half_dim 个 float

    dtype::dispatch(out->dtype, [&]<DataType D_out>() {
        using T = dtype::type_t<D_out>;
        const T* xi = static_cast<const T*>(x->data);
        T*       xo = static_cast<T*>(out->data);

        size_t head_bytes = static_cast<size_t>(D) * sizeof(T);
        size_t seq_stride = static_cast<size_t>(nh * D);
        size_t batch_stride = static_cast<size_t>(nh * S * D);

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < nh; ++h) {
                for (int64_t s = 0; s < S; ++s) {
                    // 从 cos/sin 缓存取当前 pos 的频率
                    const float* c = cos_data + s * cos_stride;
                    const float* sn = sin_data + s * cos_stride;

                    const T* row_in  = xi + b * batch_stride + h * seq_stride + s * D;
                    T*       row_out = xo + b * batch_stride + h * seq_stride + s * D;

                    for (int64_t i = 0; i < HD; ++i) {
                        float x_val = dtype::to_f32<D_out>(row_in[i]);
                        float y_val = dtype::to_f32<D_out>(row_in[i + HD]);
                        float ci    = c[i];
                        float si    = sn[i];

                        float xr = x_val * ci - y_val * si;
                        float yr = x_val * si + y_val * ci;

                        row_out[i]     = dtype::from_f32<D_out>(xr);
                        row_out[i + HD] = dtype::from_f32<D_out>(yr);
                    }
                }
            }
        }
    });
}

void rope_cache(Tensor* out) {
    // precompute cos/sin tables for RoPE
    // out->data: float [max_seq, half_dim]
    float theta     = out->op_params[0];
    int   head_dim  = static_cast<int>(out->op_params[1]);
    int   max_seq   = static_cast<int>(out->op_params[2]);
    int   half_dim  = head_dim / 2;

    bool  is_cos    = out->name.find("_cos") != std::string::npos;

    float* o = static_cast<float*>(out->data);
    for (int pos = 0; pos < max_seq; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float freq = 1.0f / std::pow(theta, 2.0f * i / head_dim);
            float angle = pos * freq;
            o[pos * half_dim + i] = is_cos ? std::cos(angle) : std::sin(angle);
        }
    }
}

} // namespace cpu
