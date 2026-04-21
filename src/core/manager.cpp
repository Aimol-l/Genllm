#include "core/manager.hpp"
#include "utils/tools.hpp"
#include <vector>
#include <print>
#include <unordered_set>
#include <cstring>

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif


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


    // 预转置 linear 权重: [out_features, in_features] → [in_features, out_features]
    // 使推理时 linear kernel 走连续内存访问路径，消除运行时 transpose 开销
    // {
    //     std::unordered_set<Tensor*> done;
    //     int count = 0;
    //     for (auto* t : graph.get_all_tensors()) {
    //         if (t->op_type != OperationType::OP_TYPE_LINEAR) continue;
    //         if (static_cast<int>(t->op_params[0]) != 1) continue;

    //         Tensor* w = t->src[1];
    //         if (!w || !w->data || done.contains(w)) continue;
    //         done.insert(w);

    //         int64_t rows = w->dims[0]; // out_features
    //         int64_t cols = w->dims[1]; // in_features
    //         size_t esz  = data_type_size(w->dtype);
    //         size_t nbytes = static_cast<size_t>(rows) * cols * esz;

    //         std::vector<uint8_t> buf(nbytes);
    //         auto* src = static_cast<const uint8_t*>(w->data);

    //         // 转置 [rows, cols] → [cols, rows]
    //         for (int64_t i = 0; i < rows; ++i) {
    //             for (int64_t j = 0; j < cols; ++j) {
    //                 std::memcpy(buf.data() + (j * rows + i) * esz,
    //                            src + (i * cols + j) * esz, esz);
    //             }
    //         }
    //         std::memcpy(const_cast<void*>(w->data), buf.data(), nbytes);

    //         // 交换维度并重算步长
    //         std::swap(w->dims[0], w->dims[1]);
    //         size_t stride = 1;
    //         for (int d = TENSOR_MAX_DIMS - 1; d >= 0; --d) {
    //             if (w->dims[d] == 0) {
    //                 w->strides[d] = 0;
    //             } else {
    //                 w->strides[d] = stride * esz;
    //                 stride *= static_cast<size_t>(w->dims[d]);
    //             }
    //         }

    //         t->op_params[0] = 0; // 标记为无需运行时转置
    //         ++count;
    //     }
    //     if (count > 0)
    //         std::println("[load_weights] pre-transposed {} linear weight tensors", count);
    // }

    this->print_all_usage();
}
