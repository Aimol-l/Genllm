
#include <vector>
#include <print>
#include <cstring>

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif

#include "core/manager.h"
#include "utils/tools.hpp"

std::unique_ptr<IMemoryResource> MemoryManager::make_resource(Device dev, size_t dev_id) {
    switch (dev) {
        case Device::CPU:
            return std::make_unique<CpuMemoryResource>(lock_memory_);
    #ifdef BACKEND_CUDA
        case Device::CUDA:
            return std::make_unique<CudaMemoryResource>(static_cast<int>(dev_id));
    #endif
        default:
            throw std::runtime_error(std::format("MemoryManager: unsupported device {}", static_cast<int>(dev)));
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
    auto res_w = this->make_resource(dev, dev_id);
    auto res_a = this->make_resource(dev, dev_id);
    DevicePools pools;
    if (weight_cap > 0) {
        pools.weight = std::make_unique<MemoryPool>(std::move(res_w), weight_cap, "weight");
    }
    if (activation_cap > 0) {
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
    for (auto& [key, pools] : devices_) {
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

    std::println("[load_weights] total {} weight tensors, {} on GPU",cpu_weights.size() + gpu_weights.size(), gpu_weights.size());

    if (!cpu_weights.empty()) {
        DevicePools* pools = this->get(Device::CPU, 0);
        if (!pools || !pools->weight) {
            throw std::runtime_error("load_weights: no CPU weight pool");
        }
        for (auto& entry : cpu_weights) {
            size_t size = entry.tensor->bytes();
            MemoryBlock block = pools->weight->allocate(size, 32);
            parser.read_tensor_data(entry.gguf_offset, block.ptr, size, entry.tensor);
            entry.tensor->data = block.ptr;
            entry.tensor->offset = block.offset;
        }
    }

#ifdef BACKEND_CUDA
    if (!gpu_weights.empty()) {
        std::vector<char> staging;
        for (auto& entry : gpu_weights) {
            size_t size = entry.tensor->bytes();
            DevicePools* pools = this->get(entry.tensor->device, 0);
            if (!pools || !pools->weight) {
                throw std::runtime_error(std::format("load_weights: no weight pool for {} tensor {}",device_to_string(entry.tensor->device), entry.tensor->name));
            }
            MemoryBlock block = pools->weight->allocate(size, 32);
            staging.resize(size);
            parser.read_tensor_data(entry.gguf_offset, staging.data(), size,entry.tensor);
            cudaMemcpy(block.ptr, staging.data(), size, cudaMemcpyHostToDevice);
            entry.tensor->data = block.ptr;
            entry.tensor->offset = block.offset;
        }
    }
#else
    if (!gpu_weights.empty()) {
        throw std::runtime_error(std::format(
            "load_weights: {} GPU weights found but CUDA backend is not enabled",
            gpu_weights.size()));
    }
#endif
    this->print_all_usage();
    LOG_INFO("Load weights done");
}
