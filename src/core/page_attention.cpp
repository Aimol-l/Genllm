#include "core/page_attention.h"
#include "utils/dtype_traits.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

BlockPool::BlockPool(void* buffer, size_t buffer_bytes, int32_t block_capacity,
                     int32_t n_kv_heads, int32_t head_dim, DataType dtype)
    : block_capacity_(block_capacity)
    , n_kv_heads_(n_kv_heads)
    , head_dim_(head_dim)
    , elem_size_(data_type_size(dtype))
    , buffer_(static_cast<uint8_t*>(buffer))
    , buffer_bytes_(buffer_bytes)
{
    block_bytes_ = static_cast<size_t>(PAGE_BLOCK_SIZE) * n_kv_heads_ * head_dim_ * elem_size_;
    free_list_.reserve(block_capacity);
}

int32_t BlockPool::alloc() {
    if (!free_list_.empty()) {
        int32_t id = free_list_.back();
        free_list_.pop_back();
        return id;
    }
    if (block_capacity_ > 0 && num_blocks_ >= block_capacity_)
        return -1;
    return num_blocks_++;
}

void BlockPool::free(int32_t block_id) {
    if (block_id >= 0 && block_id < num_blocks_)
        free_list_.push_back(block_id);
}

void* BlockPool::block_data(int32_t block_id) const {
    if (block_id < 0 || block_id >= num_blocks_ || !buffer_)
        return nullptr;
    return buffer_ + static_cast<size_t>(block_id) * block_bytes_;
}

// ---------------------------------------------------------------------------
// PagedAttentionManager
// ---------------------------------------------------------------------------

PagedAttentionManager::PagedAttentionManager(MemoryPool* pool)
    : pool_(pool)
{
}

void PagedAttentionManager::init_layer(int32_t layer_id, int32_t n_kv_heads, int32_t head_dim, DataType dtype) {
    auto& s = layers_[layer_id];
    s.n_kv_heads = n_kv_heads;
    s.head_dim = head_dim;
    s.dtype = dtype;
    s.active = true;
    s.num_cached = 0;
}

void PagedAttentionManager::reserve_layer(int32_t layer_id, int32_t max_blocks) {
    auto& s = layers_[layer_id];
    if (!s.active || !pool_) return;

    size_t block_bytes = static_cast<size_t>(PAGE_BLOCK_SIZE) * s.n_kv_heads * s.head_dim * data_type_size(s.dtype);
    size_t total_bytes = static_cast<size_t>(max_blocks) * block_bytes;

    MemoryBlock k_block = pool_->allocate(total_bytes, 64);
    s.k_pool = BlockPool(k_block.ptr, total_bytes, max_blocks, s.n_kv_heads, s.head_dim, s.dtype);

    MemoryBlock v_block = pool_->allocate(total_bytes, 64);
    s.v_pool = BlockPool(v_block.ptr, total_bytes, max_blocks, s.n_kv_heads, s.head_dim, s.dtype);
}

void PagedAttentionManager::append_kv_from_tensor(
    int32_t layer_id, const void* K_data, const void* V_data,
    int32_t n_kv_heads, int32_t Skv, int32_t head_dim, DataType dtype
) {
    auto& s = layers_[layer_id];
    if (!s.active || Skv <= 0) return;

    int32_t num_cached = s.num_cached;
    size_t elem_size = data_type_size(dtype);
    size_t head_dim_bytes = static_cast<size_t>(head_dim) * elem_size;
    size_t head_stride_bytes = static_cast<size_t>(Skv) * head_dim * elem_size;

    for (int32_t p = 0; p < Skv; ++p) {
        int32_t global_pos = num_cached + p;
        int32_t logical_block = global_pos / PAGE_BLOCK_SIZE;
        int32_t offset_in_block = global_pos % PAGE_BLOCK_SIZE;

        if (offset_in_block == 0) {
            int32_t k_id = s.k_pool.alloc();
            int32_t v_id = s.v_pool.alloc();
            if (k_id < 0 || v_id < 0) break;
            if (logical_block >= static_cast<int32_t>(s.page_table.size()))
                s.page_table.push_back({k_id, v_id});
        }

        const PageEntry& entry = s.page_table[logical_block];
        uint8_t* k_dst = static_cast<uint8_t*>(s.k_pool.block_data(entry.k_block_id))
                        + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;
        uint8_t* v_dst = static_cast<uint8_t*>(s.v_pool.block_data(entry.v_block_id))
                        + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;

        for (int32_t h = 0; h < n_kv_heads; ++h) {
            std::memcpy(k_dst + h * head_dim_bytes,
                        static_cast<const uint8_t*>(K_data) + h * head_stride_bytes + static_cast<size_t>(p) * head_dim_bytes,
                        head_dim_bytes);
            std::memcpy(v_dst + h * head_dim_bytes,
                        static_cast<const uint8_t*>(V_data) + h * head_stride_bytes + static_cast<size_t>(p) * head_dim_bytes,
                        head_dim_bytes);
        }
    }
    s.num_cached += Skv;
}

void PagedAttentionManager::append_kv_from_pos(
    int32_t layer_id, const void* K_data, const void* V_data,
    int32_t n_kv_heads, int32_t head_dim, DataType dtype,
    int32_t global_pos, int32_t count
) {
    auto& s = layers_[layer_id];
    if (!s.active) return;

    size_t elem_size = data_type_size(dtype);
    size_t head_dim_bytes = static_cast<size_t>(head_dim) * elem_size;

    for (int32_t p = 0; p < count; ++p) {
        int32_t pos = global_pos + p;
        int32_t logical_block = pos / PAGE_BLOCK_SIZE;
        int32_t offset_in_block = pos % PAGE_BLOCK_SIZE;

        if (offset_in_block == 0) {
            int32_t k_id = s.k_pool.alloc();
            int32_t v_id = s.v_pool.alloc();
            if (k_id < 0 || v_id < 0) break;
            if (logical_block >= static_cast<int32_t>(s.page_table.size()))
                s.page_table.push_back({k_id, v_id});
        }

        const PageEntry& entry = s.page_table[logical_block];
        uint8_t* k_dst = static_cast<uint8_t*>(s.k_pool.block_data(entry.k_block_id))
                        + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;
        uint8_t* v_dst = static_cast<uint8_t*>(s.v_pool.block_data(entry.v_block_id))
                        + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;

        for (int32_t h = 0; h < n_kv_heads; ++h) {
            std::memcpy(k_dst + h * head_dim_bytes,
                        static_cast<const uint8_t*>(K_data) + static_cast<size_t>(p) * head_dim_bytes,
                        head_dim_bytes);
            std::memcpy(v_dst + h * head_dim_bytes,
                        static_cast<const uint8_t*>(V_data) + static_cast<size_t>(p) * head_dim_bytes,
                        head_dim_bytes);
        }
    }
    s.num_cached = std::max(s.num_cached, global_pos + count);
}

PagedAttentionManager::LayerState& PagedAttentionManager::get_layer(int32_t layer_id) {
    return layers_[layer_id];
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
