#include <cmath>
#include <cstddef>
#include "backend/cpu/normalization.h"
#include "utils/dtype_traits.hpp"

namespace cpu {


// out = x / rms * w
// 其中 rms = sqrt(mean(x_i^2) + eps)，mean 是对 hidden_size 维度的平均
void rms_norm(Tensor* out) {
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
void layer_norm(Tensor* out) {
    const Tensor* x = out->src[0];  // [batch, seq_len, hidden_size]
    const Tensor* w = out->src[1];  // [hidden_size]
    const Tensor* b = out->src[2];  // [hidden_size] or nullptr
    float eps = out->op_params[0];

    size_t hidden_size = w->num_elements();
    size_t num_tokens = x->num_elements() / hidden_size;

    size_t x_bsz = data_type_size(x->dtype);
    size_t w_bsz = data_type_size(w->dtype);
    size_t b_bsz = b ? data_type_size(b->dtype) : 0;
    size_t o_bsz = data_type_size(out->dtype);

    const std::byte* px = static_cast<const std::byte*>(x->data);
    const std::byte* pw = static_cast<const std::byte*>(w->data);
    const std::byte* pb = b ? static_cast<const std::byte*>(b->data) : nullptr;
    std::byte*       po = static_cast<std::byte*>(out->data);

    for (size_t t = 0; t < num_tokens; ++t) {
        const std::byte* px_t = px + t * hidden_size * x_bsz;
        std::byte*       po_t = po + t * hidden_size * o_bsz;

        float mean = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i)
            mean += dtype::to_f32_rt(x->dtype, px_t + i * x_bsz);
        mean /= static_cast<float>(hidden_size);

        float var = 0.0f;
        for (size_t i = 0; i < hidden_size; ++i) {
            float d = dtype::to_f32_rt(x->dtype, px_t + i * x_bsz) - mean;
            var += d * d;
        }
        float inv_std = 1.0f / std::sqrt(var / static_cast<float>(hidden_size) + eps);

        for (size_t i = 0; i < hidden_size; ++i) {
            float fx = dtype::to_f32_rt(x->dtype, px_t + i * x_bsz);
            float fw = dtype::to_f32_rt(w->dtype, pw + i * w_bsz);
            float fb = pb ? dtype::to_f32_rt(b->dtype, pb + i * b_bsz) : 0.0f;
            dtype::from_f32_rt(out->dtype, (fx - mean) * inv_std * fw + fb, po_t + i * o_bsz);
        }
    }
}


} // namespace cpu
