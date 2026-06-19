#include <cmath>
#include <limits>
#include <vector>
#include "core/manager.h"
#include "utils/dtype_traits.hpp"
#include "backend/cpu/attention.h"

namespace ops {

    void FlashAttentionImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {

        const Tensor* Q = out->src[0];  // [batch, num_heads, seq_len_q, head_dim]      ,[1,16,4,128]
        const Tensor* K = out->src[1];  // [batch, num_kv_heads, seq_len_kv, head_dim]  ,[1,8,4,128]
        const Tensor* V = out->src[2];  // [batch, num_kv_heads, seq_len_kv, head_dim]  ,[1,8,4,128]

        int32_t batch        = static_cast<int32_t>(Q->dims[0]);
        int32_t num_q_heads  = static_cast<int32_t>(Q->dims[1]);
        int32_t seq_len      = static_cast<int32_t>(Q->dims[2]);
        int32_t n_kv_h       = static_cast<int32_t>(K->dims[1]);
        int32_t head_dim     = static_cast<int32_t>(out->op_params[0]);
        int32_t num_kv_groups = static_cast<int32_t>(out->op_params[3]);

        float scale_val = out->op_params[1];
        bool causal = static_cast<int32_t>(out->op_params[2]);

        // Q/K/V 可能是不同 dtype（如 BF16），用输出 dtype 做 dispatch
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            size_t q_sz = data_type_size(Q->dtype);
            size_t k_sz = data_type_size(K->dtype);
            size_t v_sz = data_type_size(V->dtype);
            size_t o_sz = data_type_size(out->dtype);
            // 临时 buffer: scores [seq_len]
            std::vector<float> scores(static_cast<size_t>(seq_len));
            for (int64_t b = 0; b < batch; ++b) {
                for (int64_t h = 0; h < num_q_heads; ++h) {
                    // GQA: 对应的 KV head
                    int64_t kv_h = h / num_kv_groups;
                    for (int64_t sq = 0; sq < seq_len; ++sq) {
                        size_t q_off = static_cast<size_t>((b * num_q_heads * seq_len + h * seq_len + sq) * head_dim);
                        // ── 1. 计算 scores: Q @ K^T * scale ──
                        for (int64_t skv = 0; skv < seq_len; ++skv) {
                            size_t k_off = static_cast<size_t>((b * n_kv_h * seq_len + kv_h * seq_len + skv) * head_dim);
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
                        if (causal) {
                            for (int64_t skv = sq + 1; skv < seq_len; ++skv) {
                                scores[skv] = -std::numeric_limits<float>::infinity();
                            }
                        }
                        // ── 4. Softmax ──
                        float max_val = -std::numeric_limits<float>::infinity();
                        for (int64_t skv = 0; skv < seq_len; ++skv)
                            if (scores[skv] > max_val) max_val = scores[skv];
                        float sum = 0.0f;
                        for (int64_t skv = 0; skv < seq_len; ++skv) {
                            scores[skv] = std::exp(scores[skv] - max_val);
                            sum += scores[skv];
                        }
                        for (int64_t skv = 0; skv < seq_len; ++skv)
                            scores[skv] /= sum;
                        // ── 5. scores @ V → output ──
                        size_t o_off = static_cast<size_t>((b * num_q_heads * seq_len + h * seq_len + sq) * head_dim);
                        for (int32_t d = 0; d < head_dim; ++d) {
                            float val = 0.0f;
                            for (int64_t skv = 0; skv < seq_len; ++skv) {
                                size_t v_off = static_cast<size_t>(
                                    (b * n_kv_h * seq_len + kv_h * seq_len + skv) * head_dim + d);
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

    void PagedAttentionImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        auto* mgr = g_mem_manager->get_attention_manager(out->device,0);

        int32_t layer_id = out->layer_id;
        if (!mgr || !mgr->is_active(layer_id)) return;

        Tensor* Q = out->src[0];
        Tensor* K = out->src[1];
        Tensor* V = out->src[2];

        // tensor_dump::dump_tensor_npy(Q,"dump/att_q.npy");
        // tensor_dump::dump_tensor_npy(K,"dump/att_k.npy");
        // tensor_dump::dump_tensor_npy(V,"dump/att_v.npy");

        int32_t batch = static_cast<int32_t>(Q->dims[0]);
        int32_t num_q_heads = static_cast<int32_t>(Q->dims[1]);
        int32_t seq_len = static_cast<int32_t>(Q->dims[2]);
        int32_t head_dim = static_cast<int32_t>(Q->dims[3]);

        int32_t num_kv_heads = static_cast<int32_t>(K->dims[1]);
        int32_t num_kv_groups = static_cast<int32_t>(out->op_params[3]);

        float scale = out->op_params[1];
        bool causal = static_cast<int32_t>(out->op_params[2]) != 0;

        auto& layer = mgr->get_layer(layer_id);

        mgr->append_kv_from_tensor(layer_id, K->data, V->data,num_kv_heads, seq_len, head_dim, K->dtype);

        if (!layer.active || layer.page_table.empty()) return;

        int32_t total_cached = layer.num_cached;
        int32_t num_blocks = static_cast<int32_t>(layer.page_table.size());
        int32_t block_size = PAGE_BLOCK_SIZE;
        size_t elem_size = data_type_size(Q->dtype);
        bool apply_causal = causal;

        for (int32_t b = 0; b < batch; ++b) {
            for (int32_t h = 0; h < num_q_heads; ++h) {
                int32_t kv_h = h / num_kv_groups;
                for (int32_t sq = 0; sq < seq_len; ++sq) {
                    float m = -std::numeric_limits<float>::infinity();
                    float l = 0.0f;
                    std::vector<float> o(head_dim, 0.0f);

                    int32_t q_abs = total_cached - seq_len + sq;
                    int32_t limit = apply_causal ? q_abs : (total_cached - 1);

                    for (int32_t blk = 0; blk < num_blocks; ++blk) {
                        int32_t blk_start = blk * block_size;
                        int32_t blk_end = std::min(blk_start + block_size, total_cached);
                        if (blk_start > limit) break;

                        const PageEntry& entry = layer.page_table[blk];
                        const uint8_t* k_ptr = static_cast<const uint8_t*>(layer.k_pool.block_data(entry.k_block_id));

                        const uint8_t* v_ptr = static_cast<const uint8_t*>(layer.v_pool.block_data(entry.v_block_id));
                        
                        if (!k_ptr || !v_ptr) continue;

                        for (int32_t pos_in_blk = 0; pos_in_blk < blk_end - blk_start; ++pos_in_blk) {
                            int32_t kv_pos = blk_start + pos_in_blk;
                            if (apply_causal && kv_pos > q_abs) break;

                            size_t kv_off = (static_cast<size_t>(pos_in_blk) * num_kv_heads + kv_h) * head_dim;
                            float score = 0.0f;
                            for (int32_t d = 0; d < head_dim; ++d) {
                                float qv = dtype::to_f32_rt(Q->dtype,
                                    static_cast<const uint8_t*>(Q->data)
                                    + (((static_cast<size_t>(b) * num_q_heads + h) * seq_len + sq) * head_dim + d) * elem_size);
                                float kv = dtype::to_f32_rt(Q->dtype, k_ptr + (kv_off + d) * elem_size);
                                score += qv * kv;
                            }
                            score *= scale;

                            float m_prev = m;
                            m = std::fmax(m, score);
                            float exp_score = std::exp(score - m);
                            float exp_prev = std::exp(m_prev - m);

                            for (int32_t d = 0; d < head_dim; ++d) {
                                float vv = dtype::to_f32_rt(Q->dtype, v_ptr + (kv_off + d) * elem_size);
                                o[d] = o[d] * exp_prev + exp_score * vv;
                            }
                            l = l * exp_prev + exp_score;
                        }
                    }

                    float inv_l = 1.0f / l;
                    for (int32_t d = 0; d < head_dim; ++d) {
                        uint8_t* out_ptr = static_cast<uint8_t*>(out->data)
                            + (((static_cast<size_t>(b) * num_q_heads + h) * seq_len + sq) * head_dim + d) * elem_size;
                        dtype::from_f32_rt(Q->dtype, o[d] * inv_l, out_ptr);
                    }
                }
            }
        }
    }

template struct PagedAttentionImpl<Device::CPU>;
template struct FlashAttentionImpl<Device::CPU>;
}
