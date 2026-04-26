#include "core/executor.h"
#include "core/kernels.h"
#include "model/op_factory.hpp"
#include "utils/bfloat16.hpp"
#include "utils/tools.hpp"
#include "core/page_attention.h"

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

    // 初始化 KV Cache
    auto& pam = PagedAttentionManager::instance();
    size_t kv_cache = scheduler_.config().kv_cache_per_layer;
    if (kv_cache > 0) {
        int64_t max_seq = scheduler_.config().max_seq_len;
        int32_t max_blocks = static_cast<int32_t>((max_seq + PAGE_BLOCK_SIZE - 1) / PAGE_BLOCK_SIZE);
        for (auto* t : graph_.get_all_tensors()) {
            if (t->op_type == OperationType::OP_TYPE_SDPA && t->src[1]) {
                int32_t n_kv_heads = static_cast<int32_t>(t->src[1]->dims[1]);
                int32_t head_dim = static_cast<int32_t>(t->dims[3]);
                DataType dtype = t->dtype;
                pam.init_layer(t->layer_id, n_kv_heads, head_dim, dtype);
                pam.reserve_layer(t->layer_id, max_blocks);
            }
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
    // 重新构建：按 execution_level 遍历，同一 level 的 tensor 合并，同时按 CACHE/非CACHE 分到不同 map
    persistent_map.clear();
    step_map.clear();
    for (const auto& level : graph_.get_execution_levels()) {
        // 按 layer_id + type 分桶（同一 level 内 CACHE 和 ACTIVATION 必须分开）
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
                grp.levels.push_back(std::move(tb.cache));
            }
            if (!tb.step.empty()) {
                auto& grp = step_map[lid];
                grp.layer_id = lid;
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

    // 收集所有 ApplyRoPE tensor，用于 decode 阶段更新 start_pos
    for (auto* t : graph_.get_all_tensors()) {
        if (t->op_type == OperationType::OP_TYPE_APPLY_ROPE) {
            apply_rope_tensors_.push_back(t);
        }
    }
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
                // pool_->submit([this, t]() { this->execute_tensor(t); });
                this->execute_tensor(t);
                // std::println("{}",t->name);
                // ops::println(t);
            }
            // pool_->wait();
        }
        // 非最后一层时 reset，下一层复用激活池内存
        // 跳过 layer_id == -1 (embedding): reset 会覆盖该层的输出 x_in,
        // 导致下一个 transformer 层的残差连接读取到被破坏的数据
        if (i + 1 < step_layers_.size() && step_layers_[i].layer_id != -1) {
            this->reset_step_activations();
        }
    }
}
void Executor::execute_tensor(Tensor* t) {
    if (is_view_op(t->op_type)) {
        this->execute_view(t);
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
    std::vector<int32_t> output;

    this->prefill(prompt);

    for (int i = 0; i < max_tokens; ++i) {
        int32_t next = this->sample();

        if (eos_tokens == next) break;
        output.push_back(next);

        this->decode_step(next);

        std::print("\rtokens: {}/{}",i+1,max_tokens);
        std::fflush(stdout);
    }
    std::print("\n");
    return output;
}

void Executor::prefill(const std::vector<int32_t>& token_ids) {
    this->is_prefill_ = true;
    this->seq_pos_ = 0;
    // 重置 RoPE start_pos = 0（prefill 时位置为 0,1,2,...,seq_len-1）
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
    // decode 时只处理 1 个新 token，seq_len=1
    this->resolve_dims(1, 1);
    // RoPE 偏移 = 当前序列长度（新 token 的正确位置）
    for (auto* t : apply_rope_tensors_) {
        t->op_params[2] = static_cast<float>(this->seq_pos_);
    }
    this->bind_input("input_ids", &token_id, sizeof(int32_t));
    this->forward();
    ++this->seq_pos_;
}

int32_t Executor::sample_argmax() const {
    const auto& outputs = graph_.get_external_outputs();
    if (outputs.empty() || !outputs[0]->data) {
        throw std::runtime_error("Executor::sample_argmax: output tensor not computed");
    }
    Tensor* logits = outputs[0];

    const size_t vocab_size = scheduler_.vocab_size();
    int32_t token_pos = (logits->dims[2] == 1) ? 0 : (this->seq_pos_ - 1);
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

    int32_t token_pos = (logits->dims[1] == 1) ? 0 : (this->seq_pos_ - 1);
    const bfloat16_t* logits_base = static_cast<const bfloat16_t*>(logits->data) + token_pos * vocab_size;

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
int32_t Executor::sample_top_p(float temperature, float top_p) const {
    const auto& outputs = graph_.get_external_outputs();
    if (outputs.empty() || !outputs[0]->data) {
        throw std::runtime_error("Executor::sample_top_p: output tensor not computed");
    }
    Tensor* logits = outputs[0];
    const size_t vocab_size = scheduler_.vocab_size();
    int32_t token_pos = (logits->dims[2] == 1) ? 0 : (this->seq_pos_ - 1);
    const bfloat16_t* logits_base = static_cast<const bfloat16_t*>(logits->data) + token_pos * vocab_size;

    auto probs = ops::Softmax(std::span<const bfloat16_t>(logits_base, vocab_size), temperature);

    // 按概率降序排序
    std::vector<std::pair<float, int32_t>> sorted_probs(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        sorted_probs[i] = {probs[i], static_cast<int32_t>(i)};
    }
    std::sort(sorted_probs.begin(), sorted_probs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Top-p 截断
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

    // 全分布采样
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

void Executor::dispatch_kernel(Tensor* t) {
    // auto start = std::chrono::steady_clock::now();
    switch (t->op_type) {
        case OperationType::OP_TYPE_RESHAPE:        kernel::reshape(t) ; break;
        case OperationType::OP_TYPE_VIEW:
        case OperationType::OP_TYPE_TRANSPOSE:      return;
        case OperationType::OP_TYPE_PERMUTE:        kernel::permute(t); break;
        case OperationType::OP_TYPE_MEMCPY:         kernel::memcpy(t); break;
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
