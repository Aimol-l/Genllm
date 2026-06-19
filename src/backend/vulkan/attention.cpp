
#ifdef BACKEND_VULKAN

#include <vulkan/vulkan.hpp>

#include "core/manager.h"
#include "utils/dtype_traits.hpp"
#include "core/page_attention.h"
#include "backend/vulkan/attention.h"
#include "backend/vulkan/vulkan_context.h"
#include "backend/vulkan/push_constants.h"


namespace ops {

void FlashAttentionImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    Tensor* Q = out->src[0];
    Tensor* K = out->src[1];
    Tensor* V = out->src[2];

    int32_t batch       = static_cast<int32_t>(Q->dims[0]);
    int32_t num_q_heads = static_cast<int32_t>(Q->dims[1]);
    int32_t seq_len     = static_cast<int32_t>(Q->dims[2]);
    int32_t head_dim    = static_cast<int32_t>(Q->dims[3]);
    int32_t num_kv_heads= static_cast<int32_t>(K->dims[1]);

    if (head_dim != 128)
        throw std::runtime_error("Vulkan FlashAttention requires head_dim == 128");

    auto& ctx = VulkanContext::get();
    if (!ctx.isRecording(dev_id))
        throw std::runtime_error("FlashAttentionImpl: not in recording mode");

    FlashPushConstants pc{
        .batch        = batch,
        .seq_len      = seq_len,
        .head_dim     = head_dim,
        .num_q_heads  = num_q_heads,
        .num_kv_heads = num_kv_heads,
        .group_size   = num_q_heads / num_kv_heads,
        .scale        = out->op_params[1],
    };

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            const char* name = std::is_same_v<T, float16_t> ? "flash_attn_f16" : "flash_attn_bf16";

            vk::Buffer buf_q = reinterpret_cast<VkBuffer>(Q->device_handle);
            vk::Buffer buf_k = reinterpret_cast<VkBuffer>(K->device_handle);
            vk::Buffer buf_v = reinterpret_cast<VkBuffer>(V->device_handle);
            vk::Buffer buf_o = reinterpret_cast<VkBuffer>(out->device_handle);

            vk::DescriptorBufferInfo qi(buf_q, Q->offset, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo ki(buf_k, K->offset, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo vi(buf_v, V->offset, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo oi(buf_o, out->offset, VK_WHOLE_SIZE);

            vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {qi, ki, vi, oi});

            ctx.bindPipeline(dev_id, name);
            ctx.bindDescriptorSet(dev_id, descSet);
            ctx.pushConstants(dev_id, &pc, sizeof(FlashPushConstants));

            uint32_t group_x = (static_cast<uint32_t>(batch * num_q_heads) + 3) / 4;
            uint32_t group_y = static_cast<uint32_t>(seq_len);
            ctx.dispatch(dev_id, group_x, group_y, 1);

            ctx.deferFreeDescriptorSet(dev_id, descSet);
        }
    });
}

void PagedAttentionImpl<Device::VULKAN>::execute(Tensor* out, int32_t dev_id) {
    auto* mgr = g_mem_manager->get_attention_manager(out->device, dev_id);
    int32_t layer_id = out->layer_id;
    if (!mgr || !mgr->is_active(layer_id)) return;

    Tensor* Q = out->src[0];
    Tensor* K = out->src[1];
    Tensor* V = out->src[2];

    int32_t batch      = static_cast<int32_t>(Q->dims[0]);
    int32_t num_q_heads = static_cast<int32_t>(Q->dims[1]);
    int32_t seq_len    = static_cast<int32_t>(Q->dims[2]);
    int32_t head_dim   = static_cast<int32_t>(Q->dims[3]);
    int32_t num_kv_heads= static_cast<int32_t>(K->dims[1]);

    if (head_dim != 128) throw std::runtime_error("Vulkan FlashAttention requires head_dim == 128");

    auto& ctx = VulkanContext::get();
    auto& layer = mgr->get_layer(layer_id);

    if (!ctx.isRecording(dev_id)) throw std::runtime_error("FlashAttentionImpl: not in recording mode");
    vk::CommandBuffer cmd = ctx.cmdBuffer(dev_id);

    int32_t prev_cached = layer.num_cached;
    mgr->append_kv_pages(layer_id, seq_len);
    size_t elem_sz = data_type_size(K->dtype);
    size_t hd_bytes = static_cast<size_t>(head_dim) * elem_sz;
    size_t head_stride = static_cast<size_t>(seq_len) * hd_bytes;
    size_t pool_block_bytes = static_cast<size_t>(PAGE_BLOCK_SIZE) * num_kv_heads * hd_bytes;

    vk::Buffer buf_k = reinterpret_cast<VkBuffer>(K->device_handle);
    vk::Buffer buf_v = reinterpret_cast<VkBuffer>(V->device_handle);
    vk::Buffer buf_kp = reinterpret_cast<VkBuffer>(layer.k_pool.device_handle());
    vk::Buffer buf_vp = reinterpret_cast<VkBuffer>(layer.v_pool.device_handle());

    std::vector<vk::BufferCopy> k_regions, v_regions;
    for (int32_t p = 0; p < seq_len; ++p) {
        int32_t global_pos = prev_cached + p;
        int32_t blk = global_pos / PAGE_BLOCK_SIZE;
        int32_t off = global_pos % PAGE_BLOCK_SIZE;

        const PageEntry& e = layer.page_table[blk];
        VkDeviceSize k_dst_base = static_cast<VkDeviceSize>(layer.k_pool.device_offset()) + e.k_block_id * pool_block_bytes + static_cast<VkDeviceSize>(off) * num_kv_heads * hd_bytes;
        VkDeviceSize v_dst_base = static_cast<VkDeviceSize>(layer.v_pool.device_offset()) + e.v_block_id * pool_block_bytes + static_cast<VkDeviceSize>(off) * num_kv_heads * hd_bytes;

        for (int32_t h = 0; h < num_kv_heads; ++h) {
            k_regions.push_back({K->offset + h * head_stride + static_cast<size_t>(p) * hd_bytes,
                                    k_dst_base + h * hd_bytes, hd_bytes});
            v_regions.push_back({V->offset + h * head_stride + static_cast<size_t>(p) * hd_bytes,
                                    v_dst_base + h * hd_bytes, hd_bytes});
        }
    }
    if (!k_regions.empty()) {
        cmd.copyBuffer(buf_k, buf_kp, k_regions);
        cmd.copyBuffer(buf_v, buf_vp, v_regions);
    }

    int32_t total_cached = layer.num_cached;
    int32_t num_blocks = static_cast<int32_t>(layer.page_table.size());
    if (total_cached <= 0 || num_blocks <= 0) return;

    int32_t block_stride_elems = PAGE_BLOCK_SIZE * num_kv_heads * head_dim;

    // 拷贝页表到 GPU SSBO
    std::vector<int32_t> h_page_k(num_blocks), h_page_v(num_blocks);
    for (int32_t i = 0; i < num_blocks; ++i) {
        h_page_k[i] = layer.page_table[i].k_block_id;
        h_page_v[i] = layer.page_table[i].v_block_id;
    }
    size_t pt_bytes = num_blocks * sizeof(int32_t);
    vk::DeviceMemory ptk_mem, ptv_mem;
    void *ptk_ptr = nullptr, *ptv_ptr = nullptr;
    vk::Buffer buf_ptk = ctx.createStagingBuffer(dev_id, pt_bytes, &ptk_mem, &ptk_ptr);
    vk::Buffer buf_ptv = ctx.createStagingBuffer(dev_id, pt_bytes, &ptv_mem, &ptv_ptr);
    std::memcpy(ptk_ptr, h_page_k.data(), pt_bytes);
    std::memcpy(ptv_ptr, h_page_v.data(), pt_bytes);

    vk::Buffer buf_kp_ = reinterpret_cast<VkBuffer>(layer.k_pool.device_handle());
    vk::Buffer buf_vp_ = reinterpret_cast<VkBuffer>(layer.v_pool.device_handle());

    PagedAttnPushConstants pc{
      .batch = batch,
      .seq_len          = seq_len,
      .num_q_heads     = num_q_heads,
      .num_kv_heads  = num_kv_heads,
      .total_cached = total_cached,
      .num_blocks  = num_blocks,
      .num_kv_groups = static_cast<int32_t>(out->op_params[3]),
      .head_dim    = head_dim,
      .block_stride_elems = block_stride_elems,
      .q_start     = total_cached - seq_len,
      .scale       = out->op_params[1]
    };

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>) {
            const char* name = std::is_same_v<T, float16_t> ? "page_attn_f16" : "page_attn_bf16";

            vk::Buffer buf_q = reinterpret_cast<VkBuffer>(Q->device_handle);
            vk::Buffer buf_o = reinterpret_cast<VkBuffer>(out->device_handle);

            vk::DescriptorBufferInfo qi(buf_q, Q->offset, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo oi(buf_o, out->offset, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo pki(buf_ptk, 0, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo pvi(buf_ptv, 0, VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo kpi(buf_kp_, layer.k_pool.device_offset(), VK_WHOLE_SIZE);
            vk::DescriptorBufferInfo vpi(buf_vp_, layer.v_pool.device_offset(), VK_WHOLE_SIZE);

            vk::DescriptorSet descSet = ctx.updateDescriptorSets(dev_id, name, {qi, oi, pki, pvi, kpi, vpi});

            ctx.bindPipeline(dev_id, name);
            ctx.bindDescriptorSet(dev_id, descSet);
            ctx.pushConstants(dev_id, &pc, sizeof(PagedAttnPushConstants));

            uint32_t group_y = static_cast<uint32_t>(seq_len);
            uint32_t group_x = (static_cast<uint32_t>(batch * num_q_heads) + 3) / 4;

            ctx.dispatch(dev_id, group_x, group_y, 1);

            ctx.deferFreeDescriptorSet(dev_id, descSet);
        }
    });
    auto& slot = ctx.slot(dev_id);
    slot.pending_staging_frees.emplace_back(buf_ptk, ptk_mem, ptk_ptr);
    slot.pending_staging_frees.emplace_back(buf_ptv, ptv_mem, ptv_ptr);
}

template struct PagedAttentionImpl<Device::VULKAN>;
template struct FlashAttentionImpl<Device::VULKAN>;

}

#endif
