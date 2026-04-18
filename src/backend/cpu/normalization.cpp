#include <cmath>
#include "backend/cpu/normalization.h"
#include "utils/dtype_traits.hpp"

namespace cpu {

// out = x / rms * w
// 其中 rms = sqrt(mean(x_i^2) + eps)，mean 是对 hidden_size 维度的平均
void rms_norm(Tensor* out) {

    const Tensor* x = out->src[0];  // [batch, seq_len, hidden_size]
    const Tensor* w = out->src[1];  // [hidden_size]
    float eps = out->op_params[0];
    
    // 从 weight 张量推断 hidden_size
    size_t hidden_size = w->num_elements();
    // 计算需要处理多少个 token（batch * seq_len）
    size_t num_tokens = x->num_elements() / hidden_size;
    
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        
        T*       po = static_cast<T*>(out->data);
        const T* px = static_cast<const T*>(x->data);
        const T* pw = static_cast<const T*>(w->data);
        
        // 外层循环：遍历每个 token
        for (size_t t = 0; t < num_tokens; ++t) {
            // 定位当前 token 的起始位置
            const T* px_t = px + t * hidden_size;
            T*       po_t = po + t * hidden_size;
            
            // 计算当前 token 的平方和
            float sum_sq = 0.0f;
            for (size_t i = 0; i < hidden_size; ++i) {
                float fx = dtype::to_f32<D>(px_t[i]);
                sum_sq += fx * fx;
            }
            
            // 计算 RMS
            float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
            
            // 归一化并应用权重
            for (size_t i = 0; i < hidden_size; ++i) {
                float fx = dtype::to_f32<D>(px_t[i]);
                float fw = dtype::to_f32<D>(pw[i]);
                po_t[i] = dtype::from_f32<D>(fx / rms * fw);
            }
        }
    });
}
void layer_norm(Tensor* out) {
    const Tensor* x = out->src[0];  // [batch, seq_len, hidden_size]
    const Tensor* w = out->src[1];  // [hidden_size]
    const Tensor* b = out->src[2];  // [hidden_size] or nullptr
    float eps = out->op_params[0];
    
    size_t hidden_size = w->num_elements();
    size_t num_tokens = x->num_elements() / hidden_size;
    
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* px = static_cast<const T*>(x->data);
        const T* pw = static_cast<const T*>(w->data);
        const T* pb = b ? static_cast<const T*>(b->data) : nullptr;
        T*       po = static_cast<T*>(out->data);
        
        // 外层循环：遍历每个 token
        for (size_t t = 0; t < num_tokens; ++t) {
            const T* px_t = px + t * hidden_size;
            T*       po_t = po + t * hidden_size;
            
            // 1. 计算均值
            float mean = 0.0f;
            for (size_t i = 0; i < hidden_size; ++i) {
                mean += dtype::to_f32<D>(px_t[i]);
            }
            mean /= static_cast<float>(hidden_size);
            
            // 2. 计算方差
            float var = 0.0f;
            for (size_t i = 0; i < hidden_size; ++i) {
                float d = dtype::to_f32<D>(px_t[i]) - mean;
                var += d * d;
            }
            float inv_std = 1.0f / std::sqrt(var / static_cast<float>(hidden_size) + eps);
            
            // 3. 归一化、应用权重和偏置
            for (size_t i = 0; i < hidden_size; ++i) {
                float fx = dtype::to_f32<D>(px_t[i]);
                float fw = dtype::to_f32<D>(pw[i]);
                float fb = pb ? dtype::to_f32<D>(pb[i]) : 0.0f;
                po_t[i] = dtype::from_f32<D>((fx - mean) * inv_std * fw + fb);
            }
        }
    });
}


} // namespace cpu
