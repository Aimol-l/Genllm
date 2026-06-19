#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <map>
#include "utils/utils.hpp"
#include "pools.h"

constexpr int32_t PAGE_BLOCK_SIZE = 16;

class BlockPool {
private:
    int32_t head_dim_ = 0;
    int32_t n_kv_heads_ = 0;
    int32_t num_blocks_ = 0;
    int32_t block_capacity_ = 0;

    size_t dev_handle_ = 0;
    size_t dev_offset_ = 0;
    size_t elem_size_ = 0;
    size_t block_bytes_ = 0;
    size_t buffer_bytes_ = 0;

    std::byte* buffer_ = nullptr;
    std::vector<int32_t> free_list_;
    friend class PagedAttentionManager;
public:
    BlockPool() = default;

    BlockPool(void* buffer, size_t buffer_bytes, int32_t block_capacity,int32_t n_kv_heads, int32_t head_dim, DataType dtype, size_t dev_handle = 0);

    [[nodiscard]] int32_t alloc();

    void reset();
    void free(int32_t block_id);
    
    [[nodiscard]] void* block_data(int32_t block_id) const;
    [[nodiscard]] size_t device_handle() const { return dev_handle_; }
    [[nodiscard]] size_t device_offset() const { return dev_offset_; }

    bool empty() const { return num_blocks_ == 0; }
    int32_t num_blocks() const { return num_blocks_; }
    void set_device_offset(size_t off) { dev_offset_ = off; }
    int32_t num_free() const { return static_cast<int32_t>(free_list_.size()); }
};

struct PageEntry {
    int32_t k_block_id;
    int32_t v_block_id;
};

class PagedAttentionManager {
public:
    PagedAttentionManager() = default;
    explicit PagedAttentionManager(MemoryPool* pool);

    struct LayerState {
        bool active = false;
        int32_t num_cached = 0;
        int32_t n_kv_heads = 0;
        int32_t head_dim = 0;
        DataType dtype = DataType::GGML_TYPE_F32;
        BlockPool k_pool;
        BlockPool v_pool;
        std::vector<PageEntry> page_table;
    };

    [[nodiscard]] LayerState& get_layer(int32_t layer_id);

    void reserve_layer(int32_t layer_id, int32_t max_blocks);

    void init_layer(int32_t layer_id, int32_t n_kv_heads, int32_t head_dim, DataType dtype);

    void append_kv_from_tensor(int32_t layer_id, const void* K_data, const void* V_data, int32_t n_kv_heads, int32_t seq_len, int32_t head_dim, DataType dtype);

    void append_kv_from_pos(int32_t layer_id, const void* K_data, const void* V_data, int32_t n_kv_heads, int32_t head_dim, DataType dtype, int32_t global_pos, int32_t count);

    void append_kv_pages(int32_t layer_id, int32_t count);

    void reset();

    bool is_active(int32_t layer_id) const {
        auto it = layers_.find(layer_id);
        return it != layers_.end() && it->second.active;
    }

private:
    MemoryPool* pool_ = nullptr;
    std::map<int32_t, LayerState> layers_; // layer_id -> LayerState
};