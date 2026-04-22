#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include "backend/cpu/rope.h"
#include "utils/dtype_traits.hpp"

// 对 q/k 张量应用旋转位置编码 (RoPE)
// T: x/out 的数据类型，cos/sin 为 float
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void apply_rope(
    T* out,
    const T* x,
    const float* cos_data,
    const float* sin_data,
    int64_t B,
    int64_t n_heads,
    int64_t seq_len,
    int64_t head_dim,
    int64_t half_dim,
    int64_t start_pos
){
    assert(B == 1);
    size_t head_stride = static_cast<size_t>(seq_len * head_dim);
    size_t batch_stride = static_cast<size_t>(n_heads * seq_len * head_dim);

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < n_heads; ++h) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t pos = start_pos + s;
                const float* cos = cos_data + pos * head_dim;
                const float* sin = sin_data + pos * head_dim;

                T* row_out = out + b * batch_stride + h * head_stride + s * head_dim;
                const T* row_in = x + b * batch_stride + h * head_stride + s * head_dim;

                #pragma omp simd
                for (int64_t i = 0; i < half_dim; ++i) {
                    float x0 = static_cast<float>(row_in[i]);
                    float x1 = static_cast<float>(row_in[i + half_dim]);
                    float ci = cos[i], si = sin[i];

                    row_out[i] = static_cast<T>(x0 * ci - x1 * si);
                    row_out[i + half_dim] = static_cast<T>(x0 * si + x1 * ci);
                }
            }
        }
    }
}

// out [max_seq_len, head_dim] 预计算 RoPE 的 cos/sin 表 (输出始终为 F32)
static void rope_cache(float* dst, float theta, int head_dim, int max_seq, bool is_cos) {
    int half_dim = head_dim / 2;
    std::vector<double> inv_freqs(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        double exponent = (2.0 * static_cast<double>(i)) / static_cast<double>(head_dim);
        inv_freqs[i] = 1.0 / std::pow(static_cast<double>(theta), exponent);
    }

    for (int s = 0; s < max_seq; ++s) {
        double pos = static_cast<double>(s);
        int base_idx = s * head_dim;
        for (int i = 0; i < half_dim; ++i) {
            double angle = pos * inv_freqs[i];
            float value = is_cos
                ? static_cast<float>(std::cos(angle))
                : static_cast<float>(std::sin(angle));
            dst[base_idx + i] = value;
            dst[base_idx + i + half_dim] = value;
        }
    }
}

namespace ops {

    void ApplyRopeImpl<Device::CPU>::execute(Tensor* out) {
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

    void RopeCacheImpl<Device::CPU>::execute(Tensor* out) {
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

template struct ApplyRopeImpl<Device::CPU>;
template struct RopeCacheImpl<Device::CPU>;
}
