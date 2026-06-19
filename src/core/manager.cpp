
#include "tensor.hpp"
#include "tools.hpp"
#include <vector>
#include <print>
#include <cstring>

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif
#ifdef BACKEND_VULKAN
#include "backend/vulkan/vulkan_context.h"
#endif

#include "core/manager.h"

MemoryManager* g_mem_manager = nullptr;

MemoryManager::MemoryManager()
    : lock_memory_(false) {
    g_mem_manager = this;
}

MemoryManager::MemoryManager(bool lock_memory)
    : lock_memory_(lock_memory) {
    g_mem_manager = this;
}

std::unique_ptr<IMemoryResource> MemoryManager::make_resource(Device dev, size_t dev_id) {
    switch (dev) {
        case Device::CPU:
            return std::make_unique<CpuMemoryResource>(lock_memory_);
    #ifdef BACKEND_CUDA
        case Device::CUDA:
            return std::make_unique<CudaMemoryResource>(static_cast<int>(dev_id));
    #endif
    #ifdef BACKEND_VULKAN
        case Device::VULKAN:
            return std::make_unique<VulkanMemoryResource>(static_cast<int>(dev_id));
    #endif
        default:    throw std::runtime_error(std::format("MemoryManager: unsupported device {}", static_cast<int>(dev)));
    }
}

// 创建存储管理，输入设备，权重使用量，激活使用量上限，kv-cache使用量
DevicePools& MemoryManager::get_or_create(
    Device dev, 
    size_t dev_id,
    size_t weight_cap,
    size_t activation_cap,
    size_t kv_cap)
{
    DevKey key{dev, dev_id};
    auto it = devices_.find(key);

    if (it != devices_.end()) return it->second;

    DevicePools pools;

    if (weight_cap > 0) {
        auto res_w = this->make_resource(dev, dev_id);
        pools.weight = std::make_unique<MemoryPool>(std::move(res_w), weight_cap, "weight");
    }
    if (activation_cap > 0) {
        auto res_a = this->make_resource(dev, dev_id);
        pools.activation = std::make_unique<MemoryPool>(std::move(res_a), activation_cap, "activation");
    }
    if (kv_cap > 0) {
        auto res_k = this->make_resource(dev, dev_id);
        pools.kv_cache = std::make_unique<MemoryPool>(std::move(res_k), kv_cap, "kv_cache");
    }

    auto [inserted, _] = devices_.emplace(key, std::move(pools));
    
    return inserted->second;
}

DevicePools* MemoryManager::get(Device dev, size_t dev_id) {
    DevKey key{dev, dev_id};
    auto it = devices_.find(key);
    return it != devices_.end() ? &it->second : nullptr;
}

void MemoryManager::reset_all_activations() {
    for (auto&& [key, pools] : devices_) {
        pools.reset_activation();
    }
}

void MemoryManager::print_all_usage() const {
    std::println("===================== Memory Usage =====================");
    for (const auto& [key, pools] : devices_) {
        std::println("{}:{}", device_to_string(key.dev), key.id);
        pools.print_usage();
    }
    std::println("========================================================");
}
void MemoryManager::reset_kv_cache() {
    for (auto&& [key, paged] : attention_managers_) {
        paged->reset();
    }
}
PagedAttentionManager& MemoryManager::create_attention_manager(Device dev, size_t dev_id) {
    DevKey key{dev, dev_id};
    auto it = attention_managers_.find(key);
    if (it != attention_managers_.end()) return *it->second;
    DevicePools* pools = get(dev, dev_id);
    if (!pools || !pools->kv_cache) {
        throw std::runtime_error(std::format(
            "create_attention_manager: device {}:{} has no kv_cache pool",
            device_to_string(dev), dev_id));
    }
    auto mgr = std::make_unique<PagedAttentionManager>(pools->kv_cache.get());
    auto& ref = *mgr;
    attention_managers_[key] = std::move(mgr);
    return ref;
}

void MemoryManager::load_weights(GGUFParser& parser, const ComputeGraph& graph) {
    struct WeightEntry {
        Tensor* tensor;
        uint64_t gguf_offset;
    };

    std::vector<WeightEntry> cpu_weights;
    std::vector<WeightEntry> gpu_weights;

    for (auto* t : graph.get_all_tensors()) {
        if (t->type != TensorType::TENSOR_TYPE_WEIGHT) 
            continue;
        if (t->data != nullptr) 
            continue;
        if (t->device == Device::CPU) {
            cpu_weights.push_back({t, t->offset});
        } else {
            gpu_weights.push_back({t, t->offset});
        }
    }

    std::println("Loading weights...");

    if (!cpu_weights.empty()) {
        std::println("  CPU weights: {}",cpu_weights.size());
        std::sort(cpu_weights.begin(),cpu_weights.end(),[&cpu_weights](WeightEntry&a,WeightEntry&b){
            return a.gguf_offset < b.gguf_offset;
        });

        std::vector<Tensor*> loaded;

        DevicePools* pools = this->get(Device::CPU, 0);
        if (!pools || !pools->weight) {
            throw std::runtime_error("load_weights: no CPU weight pool");
        }
        for (auto& entry : cpu_weights) {
            // 先判断一下是否为重复张量
            auto it = std::find_if(loaded.begin(),loaded.end(),[&entry](const Tensor* t) {
                return t->name == entry.tensor->name;
            });
            if (loaded.end() != it){
                entry.tensor->data = (*it)->data;
                entry.tensor->offset = (*it)->offset;
                entry.tensor->device_handle = (*it)->device_handle;
                continue;
            }
            size_t size = entry.tensor->bytes();
            MemoryBlock block = pools->weight->allocate(size, 32);
            parser.read_tensor_data(entry.gguf_offset, block.ptr, size);
            entry.tensor->data = block.ptr;
            entry.tensor->offset = block.offset;
            entry.tensor->device_handle = block.device_handle;
            loaded.push_back(entry.tensor);
        }
    }
#ifdef BACKEND_CUDA
    if (!gpu_weights.empty()) {
        std::println("  GPU weights: {}",gpu_weights.size());

        std::sort(gpu_weights.begin(),gpu_weights.end(),[](WeightEntry&a,WeightEntry&b){return a.gguf_offset < b.gguf_offset;});

        size_t max_weight = 0;
        for (auto& e : gpu_weights)
            if (e.tensor->device == Device::CUDA) max_weight = std::max(max_weight, e.tensor->bytes());
        std::vector<std::byte> staging(max_weight);

        for (auto& entry : gpu_weights) {
            if (entry.tensor->device != Device::CUDA) continue;
            int32_t dev_id = this->get_layer_device(entry.tensor->layer_id);
            size_t size = entry.tensor->bytes();
            DevicePools* pools = this->get(entry.tensor->device, dev_id);
            if (!pools || !pools->weight) {
                throw std::runtime_error(std::format("load_weights: no weight pool for {} tensor {}",device_to_string(entry.tensor->device), entry.tensor->name));
            }
            MemoryBlock block = pools->weight->allocate(size, 32);
            parser.read_tensor_data(entry.gguf_offset, staging.data(), size);
            cudaMemcpy(block.ptr, staging.data(), size, cudaMemcpyHostToDevice);
            entry.tensor->data = block.ptr;
            entry.tensor->offset = block.offset;
            entry.tensor->device_handle = block.device_handle;
        }
    }
#endif
#ifdef BACKEND_VULKAN
    if (!gpu_weights.empty()) {
        std::println("  GPU weights: {}",gpu_weights.size());

        std::sort(gpu_weights.begin(),gpu_weights.end(),[](WeightEntry&a,WeightEntry&b){return a.gguf_offset < b.gguf_offset;});

        auto& ctx = VulkanContext::get();
        // 直接遍历 gpu_weights，每个权重独立 submit（staging buffer 不能跨权重复用）
        for (auto& entry : gpu_weights) {
            if (entry.tensor->device != Device::VULKAN) continue;
            int32_t dev_id = this->get_layer_device(entry.tensor->layer_id);
            size_t size = entry.tensor->bytes();
            DevicePools* pools = this->get(Device::VULKAN, dev_id);
            if (!pools || !pools->weight) {
                throw std::runtime_error(std::format(
                    "load_weights: no weight pool for Vulkan:{} tensor {}",
                    dev_id, entry.tensor->name));
            }
            MemoryBlock block = pools->weight->allocate(size, 32);

            vk::DeviceMemory staging_mem;
            void* staging_mapped;
            vk::Buffer staging_buf = ctx.createStagingBuffer(dev_id, size, &staging_mem, &staging_mapped);

            parser.read_tensor_data(entry.gguf_offset, staging_mapped, size);

            ctx.beginRecording(dev_id);
            vk::CommandBuffer cmd = ctx.cmdBuffer(dev_id);
            vk::BufferCopy region(0, block.offset, size);
            cmd.copyBuffer(staging_buf,
                           static_cast<VkBuffer>(reinterpret_cast<void*>(block.device_handle)),
                           region);
            ctx.endRecordingAndWait(dev_id);
            ctx.destroyStagingBuffer(dev_id, staging_buf, staging_mem, staging_mapped);

            entry.tensor->data = block.ptr;
            entry.tensor->offset = block.offset;
            entry.tensor->device_handle = block.device_handle;
        }
    }
#endif
#if !defined(BACKEND_CUDA) && !defined(BACKEND_VULKAN)
    if (!gpu_weights.empty()) {
        throw std::runtime_error(std::format(
            "load_weights: {} GPU weights found but no GPU backend is enabled",
            gpu_weights.size()));
    }
#endif
    this->print_all_usage();
}
PagedAttentionManager* MemoryManager::get_attention_manager(Device dev, size_t dev_id) {
    DevKey key{dev, dev_id};
    auto it = attention_managers_.find(key);
    return it != attention_managers_.end() ? it->second.get() : nullptr;
}

