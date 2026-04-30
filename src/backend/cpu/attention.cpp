#include <cmath>
#include <limits>
#include <vector>
#include "backend/cpu/attention.h"
#include "utils/dtype_traits.hpp"
#include "core/manager.h"


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

    void SdpaImpl<Device::CPU>::execute(Tensor* out,int32_t dev_id) {
        auto* mgr = g_mem_manager->get_attention_manager(out->device,0);

        if (mgr && mgr->is_active(out->layer_id)) {
            FlashAttentionImpl<Device::CPU>::execute(out);
        } else {
            AttentionImpl<Device::CPU>::execute(out);
        }
    }
    void cpu_paged_attention(
        void* out,
        const void* Q,
        int32_t B, int32_t n_heads, int32_t Sq,
        int32_t n_kv_heads, int32_t num_kv_groups,
        int32_t head_dim,
        float scale,
        bool causal,
        DataType dtype,
        const PagedAttentionManager::LayerState& layer
    ) {
        if (!layer.active || layer.page_table.empty()) return;

        int32_t total_cached = layer.num_cached;
        int32_t num_blocks = static_cast<int32_t>(layer.page_table.size());
        int32_t block_size = PAGE_BLOCK_SIZE;
        size_t elem_size = data_type_size(dtype);
        bool apply_causal = causal && Sq > 1;

        for (int32_t b = 0; b < B; ++b) {
            for (int32_t h = 0; h < n_heads; ++h) {
                int32_t kv_h = h / num_kv_groups;
                for (int32_t sq = 0; sq < Sq; ++sq) {
                    float m = -std::numeric_limits<float>::infinity();
                    float l = 0.0f;
                    std::vector<float> o(head_dim, 0.0f);

                    int32_t limit = apply_causal ? std::min(sq, total_cached - 1) : (total_cached - 1);

                    for (int32_t blk = 0; blk < num_blocks; ++blk) {
                        int32_t blk_start = blk * block_size;
                        int32_t blk_end = std::min(blk_start + block_size, total_cached);
                        if (blk_start > limit) break;

                        const PageEntry& entry = layer.page_table[blk];
                        const uint8_t* k_ptr = static_cast<const uint8_t*>(
                            layer.k_pool.block_data(entry.k_block_id));
                        const uint8_t* v_ptr = static_cast<const uint8_t*>(
                            layer.v_pool.block_data(entry.v_block_id));
                        if (!k_ptr || !v_ptr) continue;

                        for (int32_t pos_in_blk = 0; pos_in_blk < blk_end - blk_start; ++pos_in_blk) {
                            int32_t kv_pos = blk_start + pos_in_blk;
                            if (apply_causal && kv_pos > sq) break;

                            size_t kv_off = (static_cast<size_t>(pos_in_blk) * n_kv_heads + kv_h) * head_dim;
                            float score = 0.0f;
                            for (int32_t d = 0; d < head_dim; ++d) {
                                float qv = dtype::to_f32_rt(dtype,
                                    static_cast<const uint8_t*>(Q)
                                    + (((static_cast<size_t>(b) * n_heads + h) * Sq + sq) * head_dim + d) * elem_size);
                                float kv = dtype::to_f32_rt(dtype, k_ptr + (kv_off + d) * elem_size);
                                score += qv * kv;
                            }
                            score *= scale;

                            float m_prev = m;
                            m = std::fmax(m, score);
                            float exp_score = std::exp(score - m);
                            float exp_prev = std::exp(m_prev - m);

                            for (int32_t d = 0; d < head_dim; ++d) {
                                float vv = dtype::to_f32_rt(dtype, v_ptr + (kv_off + d) * elem_size);
                                o[d] = o[d] * exp_prev + exp_score * vv;
                            }
                            l = l * exp_prev + exp_score;
                        }
                    }

                    float inv_l = 1.0f / l;
                    for (int32_t d = 0; d < head_dim; ++d) {
                        uint8_t* out_ptr = static_cast<uint8_t*>(out)
                            + (((static_cast<size_t>(b) * n_heads + h) * Sq + sq) * head_dim + d) * elem_size;
                        dtype::from_f32_rt(dtype, o[d] * inv_l, out_ptr);
                    }
                }
            }
        }
    }

    void FlashAttentionImpl<Device::CPU>::execute(Tensor* out) {
        auto* mgr = g_mem_manager->get_attention_manager(out->device,0);

        int32_t layer_id = out->layer_id;
        if (!mgr || !mgr->is_active(layer_id)) return;

        Tensor* Q = out->src[0];
        Tensor* K = out->src[1];
        Tensor* V = out->src[2];

        int32_t B = static_cast<int32_t>(Q->dims[0]);
        int32_t n_heads = static_cast<int32_t>(Q->dims[1]);
        int32_t Sq = static_cast<int32_t>(Q->dims[2]);
        int32_t head_dim = static_cast<int32_t>(Q->dims[3]);
        int32_t n_kv_heads = static_cast<int32_t>(K->dims[1]);
        int32_t Skv = static_cast<int32_t>(K->dims[2]);
        int32_t num_kv_groups = static_cast<int32_t>(out->op_params[3]);
        float scale = out->op_params[1];
        bool causal = static_cast<int32_t>(out->op_params[2]) != 0;

        auto& layer = mgr->get_layer(layer_id);

        if (Skv > 0) {
            // 保存KV到缓存池里面
            mgr->append_kv_from_tensor(layer_id, K->data, V->data,n_kv_heads, Skv, head_dim, K->dtype);
        }

        cpu_paged_attention(
            out->data, Q->data, B, n_heads, Sq,
            n_kv_heads, num_kv_groups, head_dim, scale,
            causal, Q->dtype, layer
        );
    }

template struct SoftmaxImpl<Device::CPU>;
template struct DiagMaskInfImpl<Device::CPU>;
template struct SdpaImpl<Device::CPU>;
template struct AttentionImpl<Device::CPU>;
template struct FlashAttentionImpl<Device::CPU>;
}
