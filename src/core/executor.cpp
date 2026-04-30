#include "core/executor.h"
#include "core/kernels.h"
#include "core/tokenizer.h"
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
#ifdef BACKEND_CUDA
    cudaFree(0);
    cudaDeviceSynchronize();
#endif
    for (auto* t : graph_.get_all_tensors()) {
        Device dev = t->device;
        auto dev_id = scheduler_.get_device_id(t->layer_id);
        if (dev_id_map_.contains(dev)){
            bool is_exist = false;
            for(auto& id : dev_id_map_[dev]){
                if(id == dev_id) is_exist = true;
            }
            if(is_exist) continue;
        }
        DevicePools* pools = memory_.get(dev, dev_id);
        if (pools && pools->activation) {
            dev_id_map_[dev].push_back(dev_id);
        }
    }

    // KV Cache 初始化已在 GraphScheduler::schedule() 中完成

    // 从 execution_levels_ 按 layer_id 分组：CACHE → persistent_layers_，其余 → step_layers_
    std::map<int, LayerGroup> persistent_map, step_map;
    for (const auto& level : graph_.get_execution_levels()) {
        for (Tensor* t : level) {
            if (!t->is_computed()) continue;
            auto& map = (t->type == TensorType::TENSOR_TYPE_CACHE) ? persistent_map : step_map;
            
            auto& grp = map[t->layer_id];
            grp.layer_id = t->layer_id;
            grp.device_id = scheduler_.get_device_id(t->layer_id);
            grp.levels.push_back({t});
        }
    }
    // 合并同一层内同一依赖级别的 tensor
    persistent_map.clear();
    step_map.clear();

    for (const auto& level : graph_.get_execution_levels()) {
        struct TypeBuckets { std::vector<Tensor*> cache; std::vector<Tensor*> step; };
        std::map<int, TypeBuckets> bucket;
        for (Tensor* t : level) {
            if (!t->is_computed()) continue;
            auto& b = bucket[t->layer_id];
            if (t->type == TensorType::TENSOR_TYPE_CACHE)
                b.cache.push_back(t);
            else
                b.step.push_back(t);
        }
        for (auto& [lid, tb] : bucket) {
            if (!tb.cache.empty()) {
                auto& grp = persistent_map[lid];
                grp.layer_id = lid;
                grp.device_id = scheduler_.get_device_id(lid);
                grp.levels.push_back(std::move(tb.cache));
            }
            if (!tb.step.empty()) {
                auto& grp = step_map[lid];
                grp.layer_id = lid;
                grp.device_id = scheduler_.get_device_id(lid);
                grp.levels.push_back(std::move(tb.step));
            }
        }
    }
    for (auto& [lid, grp] : persistent_map) 
        persistent_layers_.push_back(std::move(grp));
    for (auto& [lid, grp] : step_map)     
        step_layers_.push_back(std::move(grp));
    std::sort(step_layers_.begin(), step_layers_.end(),
              [](const LayerGroup& a, const LayerGroup& b) { return a.layer_id < b.layer_id; });

    for (auto* t : graph_.get_all_tensors()) {
        if (t->op_type == OperationType::OP_TYPE_APPLY_ROPE) {
            apply_rope_tensors_.push_back(t);
        }
    }
}
void Executor::forward() {
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
    // Phase 1: persistent ops（rope_cos/sin）
    if (!this->persistent_computed_) {
        for (const auto& grp : persistent_layers_) {
            for (const auto& level : grp.levels) {
                for (Tensor* t : level) {
                   this->execute_tensor(t, grp.device_id);
                }
            }
        }
        for (auto&& [dev, ids] : dev_id_map_) {
            for(int id : ids){
                DevicePools* pools = memory_.get(dev, id);
                if (pools && pools->activation) {
                    persistent_cursor_[dev] = pools->activation->used();
                }
            }
        }
        this->persistent_computed_ = true;
    }
    // Phase 2: step ops
    this->reset_step_activations();
    for (size_t i = 0; i < step_layers_.size(); ++i) {
        for (const auto& level : step_layers_[i].levels) {
            for (Tensor* t : level) {
                this->execute_tensor(t, step_layers_[i].device_id);
            }
        }
        if (i + 1 < step_layers_.size() && step_layers_[i].layer_id != -1) {
            this->reset_step_activations();
        }
    }
}
void Executor::execute_tensor(Tensor* t,int32_t dev_id) {
    if (is_view_op(t->op_type)) {
        this->execute_view(t);
        return;
    }
    this->allocate_output(t,dev_id);
    this->dispatch_kernel(t,dev_id);
#ifdef BACKEND_CUDA
    if (t->device != Device::CPU) {
        cudaDeviceSynchronize();
    }
#endif
}
// 作用：执行自回归生成，分为 prefill 和 decode 两个阶段
std::vector<int32_t> Executor::generate(
    const std::vector<int32_t>& prompt,
    int64_t max_tokens,
    int32_t eos_tokens,
    Tokenizer* tokenizer)
{
    if(prompt.empty()) throw std::invalid_argument("Executor::generate: prompt cannot be empty");
    if(max_tokens<=0) throw std::invalid_argument("Executor::generate: max_tokens must be positive");
    if(max_tokens>scheduler_.config().max_seq_len) {
        throw std::invalid_argument(std::format("Executor::generate: max_tokens {} exceeds scheduler's max_seq_len {}",max_tokens, scheduler_.config().max_seq_len));
    }
    std::vector<int32_t> output;
    this->prefill(prompt);
    for (int i = 0; i < max_tokens; ++i) {
        int32_t next = this->sample();
        if (eos_tokens == next) break;
        output.push_back(next);

        if (tokenizer) {
            std::string token_str = tokenizer->decode({next});
            std::print("{}", token_str);
            std::fflush(stdout);
        }
        this->decode_step(next);
    }
    return output;
}

void Executor::prefill(const std::vector<int32_t>& token_ids) {
    this->is_prefill_ = true;
    this->seq_pos_ = 0;
    for (auto* t : apply_rope_tensors_) {
        t->op_params[2] = 0;
    }
    this->resolve_dims(1, static_cast<int64_t>(token_ids.size()));
    this->bind_input("input_ids", const_cast<int32_t*>(token_ids.data()), token_ids.size() * sizeof(int32_t));
    this->forward();
    this->seq_pos_ = static_cast<int64_t>(token_ids.size());
}

void Executor::decode_step(int32_t token_id) {
    this->is_prefill_ = false;
    this->resolve_dims(1, 1);
    for (auto* t : apply_rope_tensors_) {
        t->op_params[2] = static_cast<float>(this->seq_pos_);
    }
    this->bind_input("input_ids", &token_id, sizeof(int32_t));
    this->forward();
    ++this->seq_pos_;
}

void Executor::append_tokens(const std::vector<int32_t>& token_ids) {
    this->is_prefill_ = true;
    int64_t n = static_cast<int64_t>(token_ids.size());
    for (auto* t : apply_rope_tensors_) {
        t->op_params[2] = static_cast<float>(this->seq_pos_);
    }
    this->resolve_dims(1, n);
    this->bind_input("input_ids", const_cast<int32_t*>(token_ids.data()), n * sizeof(int32_t));
    this->forward();
    this->seq_pos_ += n;
}

int32_t Executor::sample_argmax() const {
    const auto& outputs = graph_.get_external_outputs();
    if (outputs.empty() || !outputs[0]->data) {
        throw std::runtime_error("Executor::sample_argmax: output tensor not computed");
    }
    Tensor* logits = outputs[0];

    const size_t vocab_size = scheduler_.vocab_size();
    int32_t token_pos = logits->dims[1] - 1;
    const bfloat16_t* logits_base = static_cast<const bfloat16_t*>(logits->data) + token_pos * vocab_size;

    int32_t best = 0;
    float best_val = static_cast<float>(logits_base[0]);
    for (size_t i = 1; i < vocab_size; ++i) {
        float val = static_cast<float>(logits_base[i]);
        if (val > best_val) {
            best_val = val;
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}
int32_t Executor::sample() const {
    const auto& outputs = graph_.get_external_outputs();
    if (outputs.empty() || !outputs[0]->data) {
        throw std::runtime_error("Executor::sample: output tensor not computed");
    }
    Tensor* logits = outputs[0];

    const float top_p = scheduler_.top_p();
    const float temperature = scheduler_.temperature();
    const size_t vocab_size = scheduler_.vocab_size();

    int32_t token_pos = logits->dims[1] - 1;
    const bfloat16_t* logits_base = static_cast<const bfloat16_t*>(logits->data) + token_pos * vocab_size;

    // ========== 1. Softmax ==========
    auto probs = ops::Softmax(std::span<const bfloat16_t>(logits_base, vocab_size), temperature);

    // ========== 2. Top-p 过滤 ==========
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, int32_t>> sorted_probs;
        sorted_probs.resize(vocab_size);

        std::transform(probs.begin(), probs.end(), sorted_probs.begin(), [idx = 0](float p) mutable {
            return std::pair<float, int32_t>(p, idx++);
        });
        
        std::sort(sorted_probs.begin(), sorted_probs.end(),[](const auto& a, const auto& b) { 
            return a.first > b.first; 
        });
        
        float cumsum = 0;
        size_t cutoff = sorted_probs.size();
        for (size_t i = 0; i < sorted_probs.size(); ++i) {
            cumsum += sorted_probs[i].first;

            if (cumsum >= float(top_p)) {
                cutoff = i + 1;
                break;
            }
        }
        float truncated_sum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            truncated_sum += sorted_probs[i].first;
        }
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
int32_t Executor::sample_top_p(float temperature, float top_p) const {
    const auto& outputs = graph_.get_external_outputs();
    if (outputs.empty() || !outputs[0]->data) {
        throw std::runtime_error("Executor::sample_top_p: output tensor not computed");
    }
    Tensor* logits = outputs[0];
    const size_t vocab_size = scheduler_.vocab_size();
    int32_t token_pos = logits->dims[1] - 1;
    const bfloat16_t* logits_base = static_cast<const bfloat16_t*>(logits->data) + token_pos * vocab_size;

    auto probs = ops::Softmax(std::span<const bfloat16_t>(logits_base, vocab_size), temperature);

    std::vector<std::pair<float, int32_t>> sorted_probs(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        sorted_probs[i] = {probs[i], static_cast<int32_t>(i)};
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < vocab_size; ++i) {
            cumsum += sorted_probs[i].first;
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        float truncated_sum = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) truncated_sum += sorted_probs[i].first;

        thread_local std::mt19937 rng{std::random_device{}()};
        float rand_val = std::uniform_real_distribution<float>{0.0f, 1.0f}(rng);
        float cum_prob = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) {
            cum_prob += sorted_probs[i].first / truncated_sum;
            if (rand_val <= cum_prob) return sorted_probs[i].second;
        }
        return sorted_probs[0].second;
    }

    thread_local std::mt19937 rng{std::random_device{}()};
    float rand_val = std::uniform_real_distribution<float>{0.0f, 1.0f}(rng);
    float cum_prob = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cum_prob += probs[i];
        if (rand_val <= cum_prob) return static_cast<int32_t>(i);
    }
    return 0;
}
void Executor::bind_input(const std::string& name, void* data, size_t byte_size) {
    this->inputs_[name] = {data, byte_size};
}

void Executor::resolve_dims(int64_t batch, int64_t seq_len) {
    for (auto* t : graph_.get_all_tensors()) {
        int neg_idx = 0;

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
    for (auto&& [dev, ids] : dev_id_map_) {
        for(auto id:ids){
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
}

MemoryPool* Executor::get_act_pool(Device dev,int32_t dev_id) const {
    DevicePools* pools = memory_.get(dev, dev_id);
    return pools ? pools->activation.get() : nullptr;
}

Tensor* Executor::find_tensor(const std::string& name) const {
    for (auto* t : graph_.get_all_tensors()) {
        if (t->name == name) return t;
    }
    return nullptr;
}

void Executor::allocate_output(Tensor* t,int32_t dev_id ) {
    MemoryPool* pool = this->get_act_pool(t->device,dev_id);
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

void Executor::dispatch_kernel(Tensor* t,int32_t dev_id) {
    switch (t->op_type) {
        case OperationType::OP_TYPE_RESHAPE:        kernel::reshape(t, dev_id); break;
        case OperationType::OP_TYPE_VIEW:
        case OperationType::OP_TYPE_TRANSPOSE:      return;
        case OperationType::OP_TYPE_PERMUTE:        kernel::permute(t, dev_id); break;
        case OperationType::OP_TYPE_MEMCPY:         kernel::memcpy(t, dev_id); break;
        case OperationType::OP_TYPE_ADD:            kernel::add(t, dev_id);     break;
        case OperationType::OP_TYPE_SUB:            kernel::sub(t, dev_id);     break;
        case OperationType::OP_TYPE_MUL:            kernel::mul(t, dev_id);     break;
        case OperationType::OP_TYPE_DIV:            kernel::div(t, dev_id);     break;
        case OperationType::OP_TYPE_RMS_NORM:       kernel::rms_norm(t, dev_id);   break;
        case OperationType::OP_TYPE_LAYER_NORM:     kernel::layer_norm(t, dev_id); break;
        case OperationType::OP_TYPE_MAT_MUL:        kernel::matmul(t, dev_id);     break;
        case OperationType::OP_TYPE_LINEAR:         kernel::linear(t, dev_id);     break;
        case OperationType::OP_TYPE_SILU:           kernel::silu(t, dev_id);  break;
        case OperationType::OP_TYPE_GELU:           kernel::gelu(t, dev_id);  break;
        case OperationType::OP_TYPE_RELU:           kernel::relu(t, dev_id);  break;
        case OperationType::OP_TYPE_SOFTMAX:        kernel::softmax(t, dev_id);      break;
        case OperationType::OP_TYPE_DIAG_MASK_INF:  kernel::diag_mask_inf(t, dev_id); break;
        case OperationType::OP_TYPE_SDPA:           kernel::sdpa(t, dev_id);          break;
        case OperationType::OP_TYPE_FLASH_ATTN:     kernel::flash_attention(t, dev_id);    break;
        case OperationType::OP_TYPE_EMBEDDING:      kernel::embedding(t, dev_id);  break;
        case OperationType::OP_TYPE_APPLY_ROPE:     kernel::apply_rope(t, dev_id); break;
        case OperationType::OP_TYPE_CONCAT:         kernel::concat(t, dev_id);  break;
        case OperationType::OP_TYPE_REPEAT:         kernel::repeat(t, dev_id);  break;
        case OperationType::OP_TYPE_ROPE_CACHE:     kernel::rope_cache(t, dev_id); break;
        default:   throw std::runtime_error(std::format("Executor: unhandled op_type '{}' for tensor '{}'",operation_type_to_string(t->op_type), t->name));
    }
}
