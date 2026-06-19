
#include <algorithm>
#include <cstddef>
#include <cstring>
#include "core/page_attention.h"

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif

static void device_memcpy(void* dst, const void* src, size_t bytes, Device dev) {
    if (dev == Device::CPU) {
        std::memcpy(dst, src, bytes);
    }
#ifdef BACKEND_CUDA
    else if (dev == Device::CUDA) {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    }
#endif
    else {
        std::memcpy(dst, src, bytes);
    }
}

BlockPool::BlockPool(void* buffer, size_t buffer_bytes, int32_t block_capacity,int32_t n_kv_heads, int32_t head_dim, DataType dtype, size_t dev_handle)
    : block_capacity_(block_capacity)
    , n_kv_heads_(n_kv_heads)
    , head_dim_(head_dim)
    , dev_handle_(dev_handle)
    , elem_size_(data_type_size(dtype))
    , buffer_(static_cast<std::byte*>(buffer))
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

void BlockPool::reset() {
    free_list_.clear();
    for (int32_t i = 0; i < num_blocks_; ++i)
        free_list_.push_back(i);
}

void* BlockPool::block_data(int32_t block_id) const {
    if (block_id < 0 || block_id >= num_blocks_ || !buffer_)
        return nullptr;
    return buffer_ + static_cast<size_t>(block_id) * block_bytes_;
}


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
    auto& state = layers_[layer_id];

    if (!state.active || !pool_) return;

    // 16 * 8 * 128 * 2 = 32768 byte
    size_t block_bytes = static_cast<size_t>(PAGE_BLOCK_SIZE) * state.n_kv_heads * state.head_dim * data_type_size(state.dtype);

    size_t total_bytes = static_cast<size_t>(max_blocks) * block_bytes; // 512 * 32768 byte = 16777216

    MemoryBlock k_block = pool_->allocate(total_bytes, 32);
    MemoryBlock v_block = pool_->allocate(total_bytes, 32);

    size_t dh = pool_->device_handle();

    state.k_pool = BlockPool(k_block.ptr, total_bytes, max_blocks, state.n_kv_heads, state.head_dim, state.dtype, dh);
    state.k_pool.set_device_offset(k_block.offset);
    
    state.v_pool = BlockPool(v_block.ptr, total_bytes, max_blocks, state.n_kv_heads, state.head_dim, state.dtype, dh);
    state.v_pool.set_device_offset(v_block.offset);
}

// 把计算出来的KV数据写入到池子里面
void PagedAttentionManager::append_kv_from_tensor(
    int32_t layer_id, const void* K_data, const void* V_data,
    int32_t n_kv_heads, int32_t seq_len, int32_t head_dim, DataType dtype
) {
    auto& state = layers_[layer_id];

    if (!state.active || seq_len <= 0) return;

    Device dev = pool_ ? pool_->device() : Device::CPU;

    int32_t prev_cached = state.num_cached;

    size_t elem_size = data_type_size(dtype);
    size_t head_dim_bytes = static_cast<size_t>(head_dim) * elem_size;
    size_t head_stride_bytes = static_cast<size_t>(seq_len) * head_dim * elem_size;

    this->append_kv_pages(layer_id, seq_len);

    for (int32_t p = 0; p < seq_len; ++p) {
        int32_t global_pos = prev_cached + p;
        int32_t logical_block = global_pos / PAGE_BLOCK_SIZE;
        int32_t offset_in_block = global_pos % PAGE_BLOCK_SIZE;

        const PageEntry& entry = state.page_table[logical_block];
        uint8_t* k_dst = static_cast<uint8_t*>(state.k_pool.block_data(entry.k_block_id)) + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;
        uint8_t* v_dst = static_cast<uint8_t*>(state.v_pool.block_data(entry.v_block_id)) + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;

        for (int32_t h = 0; h < n_kv_heads; ++h) {
            device_memcpy(k_dst + h * head_dim_bytes,
                          static_cast<const uint8_t*>(K_data) + h * head_stride_bytes + static_cast<size_t>(p) * head_dim_bytes,
                          head_dim_bytes, dev);
            device_memcpy(v_dst + h * head_dim_bytes,
                          static_cast<const uint8_t*>(V_data) + h * head_stride_bytes + static_cast<size_t>(p) * head_dim_bytes,
                          head_dim_bytes, dev);
        }
    }
}
void PagedAttentionManager::append_kv_pages(int32_t layer_id, int32_t count) {
    auto& state = layers_[layer_id];
    if (!state.active || count <= 0) return;

    for (int32_t p = 0; p < count; ++p) {
        int32_t global_pos = state.num_cached + p;
        int32_t logical_block = global_pos / PAGE_BLOCK_SIZE;
        int32_t offset_in_block = global_pos % PAGE_BLOCK_SIZE;

        if (offset_in_block == 0) {

            int32_t k_id = state.k_pool.alloc();
            int32_t v_id = state.v_pool.alloc();

            if (k_id < 0 || v_id < 0) break;
            
            if (logical_block >= static_cast<int32_t>(state.page_table.size()))
                state.page_table.push_back({k_id, v_id});
        }
    }
    state.num_cached += count;
}

void PagedAttentionManager::append_kv_from_pos(
    int32_t layer_id, const void* K_data, const void* V_data,
    int32_t n_kv_heads, int32_t head_dim, DataType dtype,
    int32_t global_pos, int32_t count
) {
    auto& state = layers_[layer_id];
    if (!state.active) return;

    Device dev = pool_ ? pool_->device() : Device::CPU;
    size_t elem_size = data_type_size(dtype);
    size_t head_dim_bytes = static_cast<size_t>(head_dim) * elem_size;

    for (int32_t p = 0; p < count; ++p) {
        int32_t pos = global_pos + p;
        int32_t logical_block = pos / PAGE_BLOCK_SIZE;
        int32_t offset_in_block = pos % PAGE_BLOCK_SIZE;

        if (offset_in_block == 0) {
            int32_t k_id = state.k_pool.alloc();
            int32_t v_id = state.v_pool.alloc();
            if (k_id < 0 || v_id < 0) break;
            if (logical_block >= static_cast<int32_t>(state.page_table.size()))
                state.page_table.push_back({k_id, v_id});
        }

        const PageEntry& entry = state.page_table[logical_block];
        uint8_t* k_dst = static_cast<uint8_t*>(state.k_pool.block_data(entry.k_block_id)) + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;
        uint8_t* v_dst = static_cast<uint8_t*>(state.v_pool.block_data(entry.v_block_id)) + static_cast<size_t>(offset_in_block) * n_kv_heads * head_dim * elem_size;

        for (int32_t h = 0; h < n_kv_heads; ++h) {
            device_memcpy(k_dst + h * head_dim_bytes,
                          static_cast<const uint8_t*>(K_data) + static_cast<size_t>(p) * head_dim_bytes,
                          head_dim_bytes, dev);
            device_memcpy(v_dst + h * head_dim_bytes,
                          static_cast<const uint8_t*>(V_data) + static_cast<size_t>(p) * head_dim_bytes,
                          head_dim_bytes, dev);
        }
    }
    state.num_cached = std::max(state.num_cached, global_pos + count);
}

PagedAttentionManager::LayerState& PagedAttentionManager::get_layer(int32_t layer_id) {
    return layers_[layer_id];
}

void PagedAttentionManager::reset() {
    for (auto& [id, state] : layers_) {
        state.num_cached = 0;
        state.page_table.clear();
        state.k_pool.reset();
        state.v_pool.reset();
    }
}
