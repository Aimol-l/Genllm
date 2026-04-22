#include <cmath>
#include "backend/cpu/normalization.h"
#include "utils/dtype_traits.hpp"

// out = x / rms * w，其中 rms = sqrt(mean(x_i^2) + eps)
// T: x/out 的数据类型，w 为 float
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void rms_norm(T* out, const T* x, const float* w, size_t seq_len, size_t hidden_size, float eps) {
    for (size_t t = 0; t < seq_len; ++t) {
        const T* x_row = x + t * hidden_size;
        T* o_row = out + t * hidden_size;

        float sum_sq = 0.0f;
        float c = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float fx = static_cast<float>(x_row[i]);
            float y = fx * fx - c;
            float t_sum = sum_sq + y;
            c = (t_sum - sum_sq) - y;
            sum_sq = t_sum;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);

        for (size_t i = 0; i < hidden_size; ++i)
            o_row[i] = static_cast<T>(static_cast<float>(x_row[i]) * inv_rms * w[i]);
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
        const Tensor* x = out->src[0];  // [batch, seq_len, hidden_size],bf16
        const Tensor* w = out->src[1];  // [hidden_size]， fp32
        float eps = out->op_params[0];  // 1e-6

        size_t hidden_size = w->num_elements();
        size_t seq_len = x->num_elements() / hidden_size;
        size_t x_bsz = data_type_size(x->dtype);
        size_t w_bsz = data_type_size(w->dtype);
        size_t o_bsz = data_type_size(out->dtype);

        std::byte*       po = static_cast<std::byte*>(out->data);
        const std::byte* px = static_cast<const std::byte*>(x->data);
        const std::byte* pw = static_cast<const std::byte*>(w->data);
        for (size_t t = 0; t < seq_len; ++t) {
            const std::byte* px_t = px + t * hidden_size * x_bsz;
            std::byte*       po_t = po + t * hidden_size * o_bsz;

            float sum_sq = 0.0;
            float c = 0.0;  // Kahan compensation
            for (size_t i = 0; i < hidden_size; ++i) {
                float fx = dtype::to_f32_rt(x->dtype, px_t + i * x_bsz);
                float y = fx * fx - c;
                float t_sum = sum_sq + y;
                c = (t_sum - sum_sq) - y;
                sum_sq = t_sum;
            }
            float mean_sq = static_cast<float>(sum_sq / static_cast<float>(hidden_size));
            float inv_rms = rsqrt(mean_sq + eps);
            for (size_t i = 0; i < hidden_size; ++i) {
                float fx = dtype::to_f32_rt(x->dtype, px_t + i * x_bsz);
                float fw = dtype::to_f32_rt(w->dtype, pw + i * w_bsz);
                float val = fx * inv_rms * fw;
                dtype::from_f32_rt(out->dtype, val, po_t + i * o_bsz);
            }
        }
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
