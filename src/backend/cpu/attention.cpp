#include <cmath>
#include <limits>
#include <vector>
#include "backend/cpu/attention.h"
#include "utils/dtype_traits.hpp"


namespace ops {

    void SoftmaxImpl<Device::CPU>::execute(Tensor* out) {
        const Tensor* x = out->src[0];
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            const T* px = static_cast<const T*>(x->data);
            T*       po = static_cast<T*>(out->data);
            size_t   n  = out->num_elements();

            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < n; ++i) {
                float fx = dtype::to_f32<D>(px[i]);
                if (fx > max_val) max_val = fx;
            }

            float sum = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float fx = std::exp(dtype::to_f32<D>(px[i]) - max_val);
                sum += fx;
                po[i] = dtype::from_f32<D>(fx);
            }

            for (size_t i = 0; i < n; ++i) {
                po[i] = dtype::from_f32<D>(dtype::to_f32<D>(po[i]) / sum);
            }
        });
    }

    void DiagMaskInfImpl<Device::CPU>::execute(Tensor* out) {
        const Tensor* x = out->src[0];
        int64_t seq = x->dims[0];
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            const T* px = static_cast<const T*>(x->data);
            T*       po = static_cast<T*>(out->data);
            for (int64_t i = 0; i < seq; ++i) {
                for (int64_t j = 0; j < seq; ++j) {
                    float fx = dtype::to_f32<D>(px[i * seq + j]);
                    po[i * seq + j] = dtype::from_f32<D>(j > i ? -std::numeric_limits<float>::infinity() : fx);
                }
            }
        });
    }

    void AttentionImpl<Device::CPU>::execute(Tensor* out) {
        const Tensor* Q = out->src[0];  // [batch, num_heads, seq_len_q, head_dim]      ,[1,16,4,128]
        const Tensor* K = out->src[1];  // [batch, num_kv_heads, seq_len_kv, head_dim]  ,[1,8,4,128]
        const Tensor* V = out->src[2];  // [batch, num_kv_heads, seq_len_kv, head_dim]  ,[1,8,4,128]
        const Tensor* mask = out->src[3]; // optional attention mask [seq_len_q, seq_len_kv]
        int32_t head_dim = static_cast<int32_t>(out->op_params[0]);
        float scale_val = out->op_params[1];
        int32_t causal = static_cast<int32_t>(out->op_params[2]);
        int32_t num_kv_groups = static_cast<int32_t>(out->op_params[3]);
        int64_t B        = Q->dims[0];
        int64_t n_heads  = Q->dims[1];
        int64_t Sq       = Q->dims[2];
        int64_t n_kv_h   = K->dims[1];
        int64_t Skv      = K->dims[2];
        // Q/K/V 可能是不同 dtype（如 BF16），用输出 dtype 做 dispatch
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            size_t q_sz = data_type_size(Q->dtype);
            size_t k_sz = data_type_size(K->dtype);
            size_t v_sz = data_type_size(V->dtype);
            size_t o_sz = data_type_size(out->dtype);
            // mask（可选，F32 类型）
            const float* mask_data = mask ? static_cast<const float*>(mask->data) : nullptr;
            // 临时 buffer: scores [Skv]
            std::vector<float> scores(static_cast<size_t>(Skv));
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t h = 0; h < n_heads; ++h) {
                    // GQA: 对应的 KV head
                    int64_t kv_h = h / num_kv_groups;
                    for (int64_t sq = 0; sq < Sq; ++sq) {
                        size_t q_off = static_cast<size_t>((b * n_heads * Sq + h * Sq + sq) * head_dim);
                        // ── 1. 计算 scores: Q @ K^T * scale ──
                        for (int64_t skv = 0; skv < Skv; ++skv) {
                            size_t k_off = static_cast<size_t>((b * n_kv_h * Skv + kv_h * Skv + skv) * head_dim);
                            float dot = 0.0f;
                            for (int32_t d = 0; d < head_dim; ++d) {
                                float qv = dtype::to_f32_rt(Q->dtype,
                                    static_cast<const uint8_t*>(Q->data) + (q_off + d) * q_sz);
                                float kv = dtype::to_f32_rt(K->dtype,
                                    static_cast<const uint8_t*>(K->data) + (k_off + d) * k_sz);
                                dot += qv * kv;
                            }
                            scores[skv] = dot * scale_val;
                        }
                        // ── 2. Causal mask: skv > sq 的位置填 -inf ──
                        if (causal) {
                            for (int64_t skv = sq + 1; skv < Skv; ++skv) {
                                scores[skv] = -std::numeric_limits<float>::infinity();
                            }
                        }
                        // ── 3. 可选 mask ──
                        if (mask_data) {
                            for (int64_t skv = 0; skv < Skv; ++skv) {
                                float mv = mask_data[sq * Skv + skv];
                                if (mv == 0.0f) {
                                    scores[skv] = -std::numeric_limits<float>::infinity();
                                } else if (std::isfinite(mv)) {
                                    scores[skv] += mv;
                                }
                            }
                        }
                        // ── 4. Softmax ──
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int64_t skv = 0; skv < Skv; ++skv)
                            if (scores[skv] > max_val) max_val = scores[skv];
                        float sum = 0.0f;
                        for (int64_t skv = 0; skv < Skv; ++skv) {
                            scores[skv] = std::exp(scores[skv] - max_val);
                            sum += scores[skv];
                        }
                        for (int64_t skv = 0; skv < Skv; ++skv)
                            scores[skv] /= sum;
                        // ── 5. scores @ V → output ──
                        size_t o_off = static_cast<size_t>((b * n_heads * Sq + h * Sq + sq) * head_dim);
                        for (int32_t d = 0; d < head_dim; ++d) {
                            float val = 0.0f;
                            for (int64_t skv = 0; skv < Skv; ++skv) {
                                size_t v_off = static_cast<size_t>(
                                    (b * n_kv_h * Skv + kv_h * Skv + skv) * head_dim + d);
                                float vv = dtype::to_f32_rt(V->dtype,
                                    static_cast<const uint8_t*>(V->data) + v_off * v_sz);
                                val += scores[skv] * vv;
                            }
                            dtype::from_f32_rt(out->dtype, val,
                                static_cast<uint8_t*>(out->data) + (o_off + d) * o_sz);
                        }
                    }
                }
            }
        });
    }

    void SdpaImpl<Device::CPU>::execute(Tensor* out) {
        AttentionImpl<Device::CPU>::execute(out);
    }

    void FlashAttentionImpl<Device::CPU>::execute(Tensor* out) {
        // TODO: flash attention
    }

template struct SoftmaxImpl<Device::CPU>;
template struct DiagMaskInfImpl<Device::CPU>;
template struct SdpaImpl<Device::CPU>;
template struct AttentionImpl<Device::CPU>;
template struct FlashAttentionImpl<Device::CPU>;
}
