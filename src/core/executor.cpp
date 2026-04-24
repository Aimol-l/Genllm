#include "core/executor.h"
#include "core/kernels.h"
#include "model/op_factory.hpp"
#include "utils/bfloat16.hpp"
#include "utils/tools.hpp"

#include <cstddef>
#include <cstdint>
#include <format>
#include <print>
#include <span>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <random>

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif


Executor::Executor(GraphScheduler& scheduler)
    : scheduler_(scheduler)
    , memory_(*scheduler_.mmanager())
    , graph_(scheduler_.graph())
    , pool_(std::make_unique<ThreadPool>(std::thread::hardware_concurrency()))
{
    for (auto* t : graph_.get_all_tensors()) {
        Device dev = t->device;
        if (dev_id_map_.contains(dev))
            continue;
        DevicePools* pools = memory_.get(dev, 0);
        if (pools && pools->activation) {
            dev_id_map_[dev] = pools->activation->device_id();
        }
    }

    // 从 execution_levels_ 按 layer_id 分组：CACHE → persistent_layers_，其余 → step_layers_
    // 用 map 保持 layer_id 有序，后续转为 vector
    std::map<int, LayerGroup> persistent_map, step_map;
    for (const auto& level : graph_.get_execution_levels()) {
        for (Tensor* t : level) {
            if (!t->is_computed()) continue;
            auto& map = (t->type == TensorType::TENSOR_TYPE_CACHE) ? persistent_map : step_map;
            auto& grp = map[t->layer_id];
            grp.layer_id = t->layer_id;
            grp.levels.push_back({t});
        }
    }
    // 合并同一层内同一依赖级别的 tensor（来自同一个 execution_level 的归为同一 sub-level）
    // 上面每层只有一个 tensor per push_back，需要按 execution_level 归并
    // 重新构建：按 execution_level 遍历，同一 level + 同一 layer_id 的 tensor 合并
    persistent_map.clear();
    step_map.clear();
    for (const auto& level : graph_.get_execution_levels()) {
        // 按 layer_id 分桶
        std::map<int, std::vector<Tensor*>> bucket;
        for (Tensor* t : level) {
            if (!t->is_computed()) continue;
            auto& target_map = (t->type == TensorType::TENSOR_TYPE_CACHE) ? persistent_map : step_map;
            auto& grp = target_map[t->layer_id];
            grp.layer_id = t->layer_id;
            bucket[t->layer_id].push_back(t);
        }
        for (auto& [lid, tensors] : bucket) {
            auto is_cache = tensors.front()->type == TensorType::TENSOR_TYPE_CACHE;
            auto& target_map = is_cache ? persistent_map : step_map;
            target_map[lid].levels.push_back(std::move(tensors));
        }
    }
    for (auto& [lid, grp] : persistent_map) persistent_layers_.push_back(std::move(grp));
    for (auto& [lid, grp] : step_map)     step_layers_.push_back(std::move(grp));
    std::sort(step_layers_.begin(), step_layers_.end(),
              [](const LayerGroup& a, const LayerGroup& b) { return a.layer_id < b.layer_id; });
}
void Executor::forward() {
    // 绑定 input tensor
    for (auto* t : graph_.get_all_tensors()) {
        if (t->type != TensorType::TENSOR_TYPE_INPUT)
            continue;
        auto it = this->inputs_.find(t->name);
        if (it == this->inputs_.end()) {
            throw std::runtime_error(std::format("Executor: input tensor '{}' not bound", t->name));
        }
        t->data = it->second.data;
        t->offset = 0;
    }
    // Phase 1: 执行 persistent ops（如 rope_cos/sin），只需算一次
    if (!this->persistent_computed_) {
        for (const auto& grp : persistent_layers_) {
            for (const auto& level : grp.levels) {
                for (Tensor* t : level) {
                   this->execute_tensor(t); // 直接执行，不用使用线程池
                }
            }
        }
        // 记录各设备 activation pool 的 cursor，后续 reset 不能超过这个位置
        for (auto& [dev, id] : dev_id_map_) {
            DevicePools* pools = memory_.get(dev, id);
            if (pools && pools->activation) {
                persistent_cursor_[dev] = pools->activation->used();
            }
        }
        this->persistent_computed_ = true;
    }
    // Phase 2: 逐层执行 step ops，层间 reset 激活池实现内存复用
    this->reset_step_activations(); // 清理激活池(但不清理持久数据)
    for (size_t i = 0; i < step_layers_.size(); ++i) {
        for (const auto& level : step_layers_[i].levels) {
            for (Tensor* t : level) {
                pool_->submit([this, t]() { this->execute_tensor(t); });
            }
            pool_->wait();
        }
        // 非最后一层时 reset，下一层复用激活池内存
        if (i + 1 < step_layers_.size()) {
            this->reset_step_activations();
        }
    }
}
void Executor::execute_tensor(Tensor* t) {
    if (is_view_op(t->op_type)) {
        this->execute_view(t);
        return;
    }
    if (t->op_type == OperationType::OP_TYPE_MEMCPY) {
        this->allocate_output(t);
        this->execute_memcpy(t);
        return;
    }
    this->allocate_output(t); // 给tensor分配内存
    this->dispatch_kernel(t);
}
// 作用：执行自回归生成，分为 prefill 和 decode 两个阶段
std::vector<int32_t> Executor::generate(
    const std::vector<int32_t>& prompt,
    int64_t max_tokens,
    int32_t eos_tokens)
{
    if(prompt.empty()) throw std::invalid_argument("Executor::generate: prompt cannot be empty");
    if(max_tokens<=0) throw std::invalid_argument("Executor::generate: max_tokens must be positive");
    if(max_tokens>scheduler_.config().max_seq_len) {
        throw std::invalid_argument(std::format("Executor::generate: max_tokens {} exceeds scheduler's max_seq_len {}",max_tokens, scheduler_.config().max_seq_len));
    }
    // 构建快速查找集合
    std::vector<int32_t> output;
    std::vector<int32_t> token_cache = prompt;

    // Phase 1: Prefill
    this->prefill(prompt);

    // Phase 2: Decode 循环
    for (int i = 0; i <max_tokens; ++i) {
        int32_t next = this->sample();  // 从当前 logits 采样
        if (eos_tokens == next) break;
        output.push_back(next);
        token_cache.push_back(next);

        this->decode_step(token_cache);

        std::print("\rtokens: {}/{}",i+1,max_tokens);
        std::fflush(stdout);
    }
    std::print("\n");
    return output;
}

void Executor::prefill(const std::vector<int32_t>& token_ids) {
    this->is_prefill_ = true;
    this->seq_pos_ = 0;
    this->resolve_dims(1, static_cast<int64_t>(token_ids.size()));
    this->bind_input("input_ids", const_cast<int32_t*>(token_ids.data()), token_ids.size() * sizeof(int32_t));
    this->forward();
    this->seq_pos_ = static_cast<int64_t>(token_ids.size());
}

void Executor::decode_step(const std::vector<int32_t>& token_ids) {
    this->is_prefill_ = false;
    this->resolve_dims(1, static_cast<int64_t>(token_ids.size())); 
    this->bind_input("input_ids", const_cast<int32_t*>(token_ids.data()), token_ids.size() * sizeof(int32_t));
    this->forward();
    ++this->seq_pos_;
}
int32_t Executor::sample() const {
    Tensor* logits = find_tensor("logits"); // [1, seq_len, vocab_size]，bf16
    if (!logits || !logits->data) {
        throw std::runtime_error("Executor::sample: logits tensor not computed");
    }
    // ops::println(logits); // 调试用，打印 logits

    const float top_p = scheduler_.top_p();
    const float temperature = scheduler_.temperature();
    const size_t vocab_size = scheduler_.vocab_size();

    // 取最后一个位置的 logits
    int32_t target_seq_pos = this->seq_pos_ - 1;
    const bfloat16_t* logits_base = static_cast<const bfloat16_t*>(logits->data) + target_seq_pos * vocab_size;

    // ========== 1. Softmax ==========
    auto probs = ops::Softmax(std::span<const bfloat16_t>(logits_base, vocab_size), temperature);

    // ========== 2. Top-p 过滤 ==========
    if (top_p > 0.0f && top_p < 1.0f) {
        // 按概率降序排序
        std::vector<std::pair<float, int32_t>> sorted_probs;
        sorted_probs.resize(vocab_size);

        std::transform(probs.begin(), probs.end(), sorted_probs.begin(), [idx = 0](float p) mutable {
            return std::pair<float, int32_t>(p, idx++);
        });
        
        std::sort(sorted_probs.begin(), sorted_probs.end(),[](const auto& a, const auto& b) { 
            return a.first > b.first; 
        });
        
        // 累积概率找分界点
        float cumsum = 0;
        size_t cutoff = sorted_probs.size();
        for (size_t i = 0; i < sorted_probs.size(); ++i) {
            cumsum += sorted_probs[i].first;

            if (cumsum >= float(top_p)) {
                cutoff = i + 1;
                break;
            }
        }
        // 截断并重新归一化
        float truncated_sum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            truncated_sum += sorted_probs[i].first;
        }
        // 在截断的集合中采样
        thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<float> dist{0.0f, 1.0f};
        float rand_val = dist(rng);

        float cum_prob = 0;
        for (size_t i = 0; i < cutoff; ++i) {
            cum_prob += sorted_probs[i].first / truncated_sum;
            if (rand_val <= cum_prob) {
                return sorted_probs[i].second;
            }
        }
        return sorted_probs[0].second;
    }
    // ========== 3. 直接从全分布采样 ==========
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist{0.0f, 1.0f};
    float rand_val = dist(rng);
    float cum_prob = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cum_prob += probs[i];
        if (rand_val <= cum_prob) {
            return static_cast<int32_t>(i);
        }
    }
    return 0;
}
void Executor::bind_input(const std::string& name, void* data, size_t byte_size) {
    this->inputs_[name] = {data, byte_size};
}

// batch 无效，seq_len 由输入 prompt 决定
void Executor::resolve_dims(int64_t batch, int64_t seq_len) {
    for (auto* t : graph_.get_all_tensors()) {
        int neg_idx = 0;

        // 首次调用时保存原始 dims（含 -1），后续调用先恢复
        if (!original_dims_.contains(t->name)) {
            original_dims_[t->name] = t->dims;
        } else {
            t->dims = original_dims_[t->name];
        }

        for (int i = 0; i < TENSOR_MAX_DIMS && t->dims[i] != 0; ++i) {
            if (t->dims[i] == -1) {
                t->dims[i] = seq_len;
                ++neg_idx;
            }
        }
        if (neg_idx > 0) {
            OpFactory::compute_strides(t);
        }
    }
}

void Executor::reset_activations() {
    memory_.reset_all_activations();
}

void Executor::reset_step_activations() {
    // 只 reset persistent 之后的内存，保留 rope_cos/sin 等缓存数据
    for (auto& [dev, id] : dev_id_map_) {
        DevicePools* pools = memory_.get(dev, id);
        if (pools && pools->activation) {
            auto it = persistent_cursor_.find(dev);
            if (it != persistent_cursor_.end()) {
                pools->activation->reset_to(it->second);
            } else {
                pools->activation->reset();
            }
        }
    }
}

MemoryPool* Executor::get_act_pool(Device dev) const {
    auto it = dev_id_map_.find(dev);
    if (it == dev_id_map_.end()) return nullptr;
    DevicePools* pools = memory_.get(dev, it->second);
    return pools ? pools->activation.get() : nullptr;
}

Tensor* Executor::find_tensor(const std::string& name) const {
    for (auto* t : graph_.get_all_tensors()) {
        if (t->name == name) return t;
    }
    return nullptr;
}

void Executor::allocate_output(Tensor* t) {
    MemoryPool* pool = get_act_pool(t->device);
    if (!pool) {
        throw std::runtime_error(std::format("Executor: no activation pool for {} tensor '{}'",device_to_string(t->device), t->name));
    }
    size_t nbytes = t->bytes();
    if (nbytes == 0) {
        throw std::runtime_error(std::format("Executor: tensor '{}' has 0 bytes (unresolved dims?)", t->name));
    }
    MemoryBlock block = pool->allocate(nbytes, 64);
    t->data = block.ptr;
    t->offset = block.offset;
}

void Executor::execute_view(Tensor* t) {
    Tensor* src = t->src[0];
    if (!src || !src->data) {
        throw std::runtime_error(std::format(
            "Executor::view: source of '{}' has no data", t->name));
    }
    t->data = src->data;
    t->offset = src->offset;
}

void Executor::execute_memcpy(Tensor* t) {
    Tensor* src = t->src[0];
    if (!src || !src->data) {
        throw std::runtime_error(std::format(
            "Executor::memcpy: source of '{}' has no data", t->name));
    }
    size_t nbytes = t->bytes();
    Device src_dev = src->device;
    Device dst_dev = t->device;

    if (src_dev == dst_dev) {
        std::memcpy(t->data, src->data, nbytes);
        return;
    }

#ifdef BACKEND_CUDA
    if (src_dev == Device::CPU && dst_dev == Device::CUDA) {
        cudaMemcpy(t->data, src->data, nbytes, cudaMemcpyHostToDevice);
    } else if (src_dev == Device::CUDA && dst_dev == Device::CPU) {
        cudaMemcpy(t->data, src->data, nbytes, cudaMemcpyDeviceToHost);
    } else {
        throw std::runtime_error(std::format(
            "Executor::memcpy: unsupported {} -> {}",
            device_to_string(src_dev), device_to_string(dst_dev)));
    }
#else
    throw std::runtime_error(std::format(
        "Executor::memcpy: cross-device copy requires CUDA backend ({} -> {})",
        device_to_string(src_dev), device_to_string(dst_dev)));
#endif
}

void Executor::dispatch_kernel(Tensor* t) {
    // auto start = std::chrono::steady_clock::now();
    switch (t->op_type) {
        case OperationType::OP_TYPE_RESHAPE:        kernel::reshape(t) ; break;
        case OperationType::OP_TYPE_VIEW:
        case OperationType::OP_TYPE_TRANSPOSE:      return;
        case OperationType::OP_TYPE_PERMUTE:        kernel::permute(t); break;
        case OperationType::OP_TYPE_MEMCPY:         return;
        case OperationType::OP_TYPE_ADD:            kernel::add(t);     break;
        case OperationType::OP_TYPE_SUB:            kernel::sub(t);     break;
        case OperationType::OP_TYPE_MUL:            kernel::mul(t);     break;
        case OperationType::OP_TYPE_DIV:            kernel::div(t);     break;
        case OperationType::OP_TYPE_RMS_NORM:       kernel::rms_norm(t);   break;
        case OperationType::OP_TYPE_LAYER_NORM:     kernel::layer_norm(t); break;
        case OperationType::OP_TYPE_MAT_MUL:        kernel::matmul(t);     break;
        case OperationType::OP_TYPE_LINEAR:         kernel::linear(t);     break;
        case OperationType::OP_TYPE_SILU:           kernel::silu(t);  break;
        case OperationType::OP_TYPE_GELU:           kernel::gelu(t);  break;
        case OperationType::OP_TYPE_RELU:           kernel::relu(t);  break;
        case OperationType::OP_TYPE_SOFTMAX:        kernel::softmax(t);      break;
        case OperationType::OP_TYPE_DIAG_MASK_INF:  kernel::diag_mask_inf(t); break;
        case OperationType::OP_TYPE_SDPA:           kernel::sdpa(t);          break;
        case OperationType::OP_TYPE_FLASH_ATTN:     kernel::flash_attention(t);    break;
        case OperationType::OP_TYPE_EMBEDDING:      kernel::embedding(t);  break;
        case OperationType::OP_TYPE_APPLY_ROPE:     kernel::apply_rope(t); break;
        case OperationType::OP_TYPE_CONCAT:         kernel::concat(t);  break;
        case OperationType::OP_TYPE_REPEAT:         kernel::repeat(t);  break;
        case OperationType::OP_TYPE_ROPE_CACHE:     kernel::rope_cache(t); break;
        default:   throw std::runtime_error(std::format("Executor: unhandled op_type '{}' for tensor '{}'",operation_type_to_string(t->op_type), t->name));
    }
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double, std::milli> duration = end - start;
    // static std::mutex log_mutex;
    // {
    //     std::lock_guard<std::mutex> lock(log_mutex);
    //     std::println("{:<13} {:<20} {:<16} {:>10.1f}ms", operation_type_to_string(t->op_type), t->name, t->dims, duration.count());
    // }
}
