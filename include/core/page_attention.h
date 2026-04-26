#pragma once
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory>
#include "utils/utils.hpp"
#include "utils/dtype_traits.hpp"

constexpr int32_t PAGE_BLOCK_SIZE = 16;

struct PageEntry {
    int32_t k_block_id;
    int32_t v_block_id;
};

class BlockPool {
public:
    BlockPool() = default;
    BlockPool(int32_t block_capacity, int32_t n_kv_heads, int32_t head_dim, DataType dtype);

    int32_t alloc();
    void free(int32_t block_id);
    void* block_data(int32_t block_id) const;
    int32_t num_blocks() const { return static_cast<int32_t>(blocks_.size()); }
    int32_t num_free() const { return static_cast<int32_t>(free_list_.size()); }
    bool empty() const { return blocks_.empty(); }

private:
    int32_t block_capacity_ = 0;
    int32_t n_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    size_t elem_size_ = 0;
    std::vector<std::vector<uint8_t>> blocks_;
    std::vector<int32_t> free_list_;

    friend class PagedAttentionManager;
};

class PagedAttentionManager {
public:
    static PagedAttentionManager& instance();

    struct LayerState {
        int32_t n_kv_heads = 0;
        int32_t head_dim = 0;
        DataType dtype = DataType::GGML_TYPE_F32;
        bool active = false;

        BlockPool k_pool;
        BlockPool v_pool;
        std::vector<PageEntry> page_table;
        int32_t num_cached = 0;
    };

    void init_layer(int32_t layer_id, int32_t n_kv_heads, int32_t head_dim, DataType dtype);
    void reserve_layer(int32_t layer_id, int32_t max_blocks);

    void append_kv_from_tensor(int32_t layer_id, const void* K_data, const void* V_data,
                               int32_t n_kv_heads, int32_t Skv, int32_t head_dim, DataType dtype);

    void append_kv_from_pos(int32_t layer_id, const void* K_data, const void* V_data,
                            int32_t n_kv_heads, int32_t head_dim, DataType dtype,
                            int32_t global_pos, int32_t count);

    LayerState& get_layer(int32_t layer_id);

    bool is_active(int32_t layer_id) const {
        auto it = layers_.find(layer_id);
        return it != layers_.end() && it->second.active;
    }

private:
    std::unordered_map<int32_t, LayerState> layers_;
};

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
);
