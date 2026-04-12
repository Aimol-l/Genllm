#include "core/executor.h"
#include "core/kernels.h"
#include "model/op_factory.hpp"
#include <format>
#include <stdexcept>
#include <cstring>

#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif


Executor::Executor(GraphScheduler& scheduler)
    : scheduler_(scheduler)
    , memory_(*scheduler_.mmanager())
    , graph_(scheduler_.graph())
{
    for (auto* t : graph_.get_all_tensors()) {
        Device dev = t->device;
        if (dev_id_map_.contains(dev)) continue;
        DevicePools* pools = memory_.get(dev, 0);
        if (pools && pools->activation) {
            dev_id_map_[dev] = pools->activation->device_id();
        }
    }
}

// 作用：执行自回归生成，分为 prefill 和 decode 两个阶段
std::vector<int32_t> Executor::generate(
    const std::vector<int32_t>& prompt,
    int max_tokens,
    int eos_token)
{
    std::vector<int32_t> output;
    // Phase 1: Prefill — 处理整个 prompt
    this->prefill(prompt);

    // Phase 2: Decode 循环 — 逐 token 生成
    for (int i = 0; i < max_tokens; ++i) {
        int32_t next = this->sample();
        if (next == eos_token) 
            break;
        output.push_back(next);
        this->decode_step(next);
    }
    return output;
}

// 
void Executor::prefill(const std::vector<int32_t>& token_ids) {
    is_prefill_ = true;
    seq_pos_ = 0;

    // 设定输入维度（batch=1, seq_len=token_ids.size()）
    this->resolve_dims(1, static_cast<int64_t>(token_ids.size()));

    // 绑定输入数据
    this->bind_input("input_ids",const_cast<int32_t*>(token_ids.data()),token_ids.size() * sizeof(int32_t));

    this->forward();

    seq_pos_ = static_cast<int64_t>(token_ids.size());
}

// ========== Decode ==========

int32_t Executor::decode_step(int32_t token) {
    is_prefill_ = false;

    // TODO: decode 时只重置激活池，不重置 KV cache 池
    // 目前 reset_all_activations 会重置所有激活池（包括 KV cache）
    // 等 KV cache 实现后需要改为只重置 activation pool
    this->reset_activations();

    // decode 阶段 seq_len = 1
    this->resolve_dims(1, 1);
    this->bind_input("input_ids", &token, sizeof(int32_t));

    this->forward();

    ++seq_pos_;
    return sample();
}

// ========== Sampling (stub) ==========

int32_t Executor::sample() const {
    Tensor* logits = find_tensor("logits");
    if (!logits || !logits->data) {
        throw std::runtime_error("Executor::sample: logits tensor not computed");
    }
    // TODO: 实现 argmax / top-k / top-p 采样
    // 目前返回 stub 值
    return -1;
}

// ========== 底层执行 ==========

void Executor::forward() {
    this->reset_activations(); // 每次 forward 前重置激活池，确保没有过期数据

    // 绑定 input tensor
    for (auto* t : graph_.get_all_tensors()) {
        if (t->type != TensorType::TENSOR_TYPE_INPUT) continue;
        auto it = inputs_.find(t->name);
        if (it == inputs_.end()) {
            throw std::runtime_error(std::format("Executor: input tensor '{}' not bound", t->name));
        }
        t->data = it->second.data;
        t->offset = 0;
    }

    // 遍历 execution_order
    for (Tensor* t : graph_.get_execution_order()) {
        if (is_view_op(t->op_type)) {
            this->execute_view(t);
            continue;
        }
        if (t->op_type == OperationType::OP_TYPE_MEMCPY) {
            this->allocate_output(t);
            this->execute_memcpy(t);
            continue;
        }
        this->allocate_output(t);
        this->dispatch_kernel(t);
    }
}

// ========== 辅助方法 ==========

void Executor::bind_input(const std::string& name, void* data, size_t byte_size) {
    inputs_[name] = {data, byte_size};
}

void Executor::resolve_dims(int64_t batch, int64_t seq_len) {
    for (auto* t : graph_.get_all_tensors()) {
        int neg_idx = 0;
        for (int i = 0; i < TENSOR_MAX_DIMS && t->dims[i] != 0; ++i) {
            if (t->dims[i] == -1) {
                t->dims[i] = (neg_idx == 0) ? batch : seq_len;
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
        throw std::runtime_error(std::format(
            "Executor: no activation pool for {} tensor '{}'",
            device_to_string(t->device), t->name));
    }
    size_t nbytes = t->bytes();
    if (nbytes == 0) {
        throw std::runtime_error(std::format(
            "Executor: tensor '{}' has 0 bytes (unresolved dims?)", t->name));
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
    switch (t->op_type) {
        case OperationType::OP_TYPE_RESHAPE:
        case OperationType::OP_TYPE_PERMUTE:
        case OperationType::OP_TYPE_VIEW:
        case OperationType::OP_TYPE_TRANSPOSE:  return;
        case OperationType::OP_TYPE_MEMCPY:     return;
        case OperationType::OP_TYPE_ADD:     kernel::add(t);     break;
        case OperationType::OP_TYPE_SUB:     kernel::sub(t);     break;
        case OperationType::OP_TYPE_MUL:     kernel::mul(t);     break;
        case OperationType::OP_TYPE_DIV:     kernel::div(t);     break;
        case OperationType::OP_TYPE_SCALE:   kernel::scale(t);   break;
        case OperationType::OP_TYPE_RMS_NORM:   kernel::rms_norm(t);   break;
        case OperationType::OP_TYPE_LAYER_NORM: kernel::layer_norm(t); break;
        case OperationType::OP_TYPE_MAT_MUL:   kernel::matmul(t);     break;
        case OperationType::OP_TYPE_LINEAR:    kernel::linear(t);     break;
        case OperationType::OP_TYPE_SILU:  kernel::silu(t);  break;
        case OperationType::OP_TYPE_GELU:  kernel::gelu(t);  break;
        case OperationType::OP_TYPE_RELU:  kernel::relu(t);  break;
        case OperationType::OP_TYPE_SOFTMAX:      kernel::softmax(t);      break;
        case OperationType::OP_TYPE_DIAG_MASK_INF: kernel::diag_mask_inf(t); break;
        case OperationType::OP_TYPE_SDPA:          kernel::sdpa(t);          break;
        case OperationType::OP_TYPE_FLASH_ATTN:    kernel::flash_attn(t);    break;
        case OperationType::OP_TYPE_EMBEDDING:  kernel::embedding(t);  break;
        case OperationType::OP_TYPE_APPLY_ROPE: kernel::apply_rope(t); break;
        case OperationType::OP_TYPE_GET_ROWS:   kernel::get_rows(t);   break;
        case OperationType::OP_TYPE_CONCAT:  kernel::concat(t);  break;
        case OperationType::OP_TYPE_REPEAT:  kernel::repeat(t);  break;
        case OperationType::OP_TYPE_ROPE_CACHE: kernel::rope_cache(t); break;
        case OperationType::OP_TYPE_SAMPLING:   kernel::sampling(t);   break;
        default:
            throw std::runtime_error(std::format("Executor: unhandled op_type '{}' for tensor '{}'",operation_type_to_string(t->op_type), t->name));
    }
}
