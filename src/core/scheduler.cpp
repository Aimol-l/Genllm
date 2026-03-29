#include "backend/backend.h"
#include "core/scheduler.h"

MemoryPool* GraphScheduler::get_pool(Device dev) {
    auto it = pools_.find(dev);
    return (it != pools_.end()) ? it->second.get() : nullptr;
}

void GraphScheduler::schedule(ComputeGraph& graph, const std::vector<BackendInfo>& devices){
    if (devices.empty()) {
        throw std::runtime_error("No backend devices provided");
    }
    std::println("=== Graph Scheduler Started ===");
    for (const auto& dev : devices) {
        std::println(
            " {} - {} GB (reserved {} GB)", 
            device_to_string(dev.device),
            dev.total_memory / 1024 / 1024 / 1024,
            dev.reserved_memory / 1024 / 1024 / 1024
        );
    }
    
    // Step 1: 提取算子特征
    auto op_features = extract_op_features(graph);
    std::println("Extracted {} compute ops", op_features.size());
    
    // Step 2: 初始化设备状态
    DeviceState dev_state = init_device_state(devices);
    
    // Step 3: 初始化内存池
    init_memory_pools(devices);
    
    // Step 4: 预处理算子强制分配 CPU
    assign_cpu_only_ops(op_features, dev_state);
    std::println("Assigned {} CPU-only ops", config_.cpu_only_ops.size());
    
    // Step 5: 贪心分配剩余算子（按拓扑序）
    for (Tensor* t : graph.get_execution_order()) {
        if (!op_features.count(t)) continue;
        greedy_assign(op_features[t], dev_state, op_features);
    }
    std::println("Greedy assignment complete");
    
    // Step 6: 插入跨设备拷贝边
    insert_copy_edges(graph, op_features, dev_state);
    std::println("Inserted cross-device copy edges");
    
    // Step 7: 强制外部输出回到 CPU
    finalize_external_outputs(graph, dev_state, op_features);
    std::println("Finalized external outputs (CPU)");
    
    // Step 8: 验证 + 回退（显存不足时迁移）
    if (!validate_and_rebalance(graph, dev_state, op_features)) {
        std::println("Warning: Schedule may cause OOM");
    }
    
    // Step 9: 分配内存池偏移
    allocate_memory(graph);
    std::println("Memory allocated");
    
    // 打印统计
    print_statistics(dev_state);
    std::println("=== Graph Scheduler Complete ===");
}

std::unordered_map<Tensor*, GraphScheduler::OpFeature> GraphScheduler::extract_op_features(const ComputeGraph& graph){
    std::unordered_map<Tensor*, OpFeature> features;
    for (Tensor* t : graph.get_execution_order()) {
        OpFeature f;
        f.tensor = t;
        f.op_type = t->op_type;
        f.flops = estimate_flops(t);
        
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            Tensor* src = t->src[i];
            if (!src) continue;
            
            size_t bytes = src->bytes();
            if (src->type == TensorType::TENSOR_TYPE_WEIGHT) {
                f.weight_bytes += bytes;
            } else {
                f.bytes_read += bytes;
                f.input_tensors.push_back(src);
            }
        }
        f.bytes_write = t->bytes();
        f.compute_intensity = static_cast<double>(f.flops) / std::max(int64_t(1), f.bytes_read + f.bytes_write);
        features[t] = f;
    }
    return features;
}

GraphScheduler::DeviceState GraphScheduler::init_device_state(const std::vector<BackendInfo>& devices) {
    DeviceState state;
    for (const auto& dev : devices) {
        state.info[dev.device] = dev;
        state.stats[dev.device] = {};
    }
    return state;
}

void GraphScheduler::init_memory_pools(const std::vector<BackendInfo>& devices) {
    for (const auto& dev : devices) {
        pools_[dev.device] = std::make_unique<MemoryPool>(
            dev.device, dev.total_memory, dev.id);
    }
}
void GraphScheduler::assign_cpu_only_ops(
    const std::unordered_map<Tensor*, OpFeature>& features,
    DeviceState& dev_state)
{
    for (const auto& [t, f] : features) {
        if (config_.cpu_only_ops.count(f.op_type)) {
            assign_to_device(t, Device::CPU, f, dev_state);
        }
    }
}

void GraphScheduler::greedy_assign(
    OpFeature& op, DeviceState& dev_state,
    const std::unordered_map<Tensor*, OpFeature>& all_features)
{
    // 用户偏好优先
    if (op.preferred_device != Device::AUTO && 
        dev_state.available_memory(op.preferred_device) >= required_memory(op)) {
        assign_to_device(op.tensor, op.preferred_device, op, dev_state);
        return;
    }
    // 计算每个设备的评分
    Device best_device = Device::CPU;
    double best_score = -1e18;
    for (const auto& [dev, info] : dev_state.info) {
        if (dev_state.available_memory(dev) < required_memory(op)) {
            continue;
        }
        double score = compute_score(op, dev, dev_state, all_features);
        if (score > best_score) {
            best_score = score;
            best_device = dev;
        }
    }
    assign_to_device(op.tensor, best_device, op, dev_state);
}
    
double GraphScheduler::compute_score(
    const OpFeature& op, Device dev, 
    const DeviceState& dev_state,
    const std::unordered_map<Tensor*, OpFeature>& all_features)
{
    double score = 0.0;
    // (1) 计算收益
    if (op.compute_intensity > config_.compute_intensity_threshold) {
        score += config_.weight_compute * op.flops * dev_state.info.at(dev).compute_power;
    }
    // (2) 通信开销
    int64_t cross_device_bytes = 0;
    for (Tensor* input : op.input_tensors) {
        if (input->device != dev) {
            cross_device_bytes += input->bytes();
        }
    }
    if (cross_device_bytes > static_cast<int64_t>(config_.small_tensor_threshold)) {
        double comm_cost = static_cast<double>(cross_device_bytes) / 
                            dev_state.info.at(dev).bandwidth / 1024 / 1024 / 1024;
        score -= config_.weight_comm * comm_cost;
    }
    
    // (3) ★ 设备亲和性（防止 Ping-pong）
    Tensor* main_input = nullptr;
    size_t main_input_bytes = 0;
    for (Tensor* input : op.input_tensors) {
        size_t bytes = input->bytes();
        if (bytes > main_input_bytes) {
            main_input_bytes = bytes;
            main_input = input;
        }
    }
    if (main_input && main_input->device == dev) {
        double affinity_bonus = config_.weight_comm * 
                                (static_cast<double>(main_input_bytes) / 
                                dev_state.info.at(dev).bandwidth / 1024 / 1024 / 1024) * 10.0;
        score += affinity_bonus;
    }
    
    // (4) 显存压力
    double mem_ratio = static_cast<double>(dev_state.available_memory(dev)) / 
                        dev_state.info.at(dev).available_memory();
    double mem_penalty = (1.0 - mem_ratio) * (1.0 - mem_ratio);
    score -= config_.weight_memory * mem_penalty * op.weight_bytes;
    
    // (5) 负载均衡
    double load = static_cast<double>(dev_state.stats.at(dev).assigned_flops) / 
                    dev_state.info.at(dev).compute_power;
    score += config_.weight_balance * (1.0 / (1.0 + load));
    
    return score;
}

void GraphScheduler::assign_to_device(Tensor* t, Device dev, const OpFeature& op, DeviceState& dev_state) {
    t->device = dev;
    auto& stats = dev_state.stats[dev];
    stats.used_memory += required_memory(op);
    stats.assigned_flops += op.flops;
    stats.assigned_ops += 1;
    if (op.weight_bytes > 0) {
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            Tensor* src = t->src[i];
            if (src && src->type == TensorType::TENSOR_TYPE_WEIGHT && 
                src->device == Device::AUTO) {
                src->device = dev;
                dev_state.stats[dev].used_memory += src->bytes();
            }
        }
    }
}

void GraphScheduler::insert_copy_edges(
    ComputeGraph& graph, 
    std::unordered_map<Tensor*, OpFeature>& features,
    DeviceState& dev_state)
{
    for (Tensor* consumer : graph.get_execution_order()) {
        if (!consumer->is_computed()) continue;
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            Tensor* src = consumer->src[i];
            if (!src || src->device == consumer->device) continue;
            Tensor* proxy = create_memcpy_proxy(src, consumer->device, dev_state);
            consumer->src[i] = proxy;
            // 将拷贝节点加入特征表（确保被执行）
            OpFeature copy_feature;
            copy_feature.tensor = proxy;
            copy_feature.op_type = OperationType::OP_TYPE_MEMCPY;
            copy_feature.bytes_read = src->bytes();
            copy_feature.bytes_write = proxy->bytes();
            features[proxy] = copy_feature;
        }
    }
}

Tensor* GraphScheduler::create_memcpy_proxy(Tensor* src, Device dst_dev, DeviceState& dev_state) {
    auto* proxy = new Tensor();
    proxy->name = "memcpy_" + src->name + "_to_" + device_to_string(dst_dev);
    proxy->op_type = OperationType::OP_TYPE_MEMCPY;
    proxy->src[0] = src;
    proxy->device = dst_dev;
    proxy->type = TensorType::TENSOR_TYPE_TEMP;
    proxy->dtype = src->dtype;
    proxy->dims = src->dims;
    
    // 预分配内存
    auto* pool = pools_[dst_dev].get();
    proxy->offset = pool->allocate(proxy->bytes());
    proxy->data = pool->get_ptr(proxy->offset);
    proxy->backend = pool;
    
    // 更新设备状态
    dev_state.stats[dst_dev].used_memory += proxy->bytes();
    
    return proxy;
}
void GraphScheduler::finalize_external_outputs(
    ComputeGraph& graph, DeviceState& dev_state,
    std::unordered_map<Tensor*, OpFeature>& features)
{
    for (Tensor* out : graph.get_external_outputs()) {
        if (!out || out->device == Device::CPU) continue;
        std::println("Output '{}' on {}, inserting D2H copy", out->name, device_to_string(out->device));
        Tensor* cpu_proxy = create_memcpy_proxy(out, Device::CPU, dev_state);
        graph.replace_output(out, cpu_proxy);
        OpFeature copy_feature;
        copy_feature.tensor = cpu_proxy;
        copy_feature.op_type = OperationType::OP_TYPE_MEMCPY;
        features[cpu_proxy] = copy_feature;
    }
}  

bool GraphScheduler::validate_and_rebalance(
    ComputeGraph& graph, DeviceState& dev_state,
    std::unordered_map<Tensor*, OpFeature>& features)
{
    for (const auto& [dev, info] : dev_state.info) {
        if (dev_state.stats[dev].used_memory > info.available_memory()) {
            std::println("Warning: {} memory overflow, rebalancing...", device_to_string(dev));
            rebalance_to_less_loaded(graph, dev_state, features);
            return false;
        }
    }
    return true;
}

void GraphScheduler::rebalance_to_less_loaded(
    ComputeGraph& graph, DeviceState& dev_state,
    std::unordered_map<Tensor*, OpFeature>& features)
{
    std::vector<std::pair<Tensor*, double>> candidates;
    for (const auto& [t, f] : features) {
        if (t->device == Device::CPU || t->op_type == OperationType::OP_TYPE_MEMCPY) continue;
        
        double benefit = t->bytes() * (1.0 / std::max(1.0, f.compute_intensity));
        candidates.emplace_back(t, benefit);
    }
    std::sort(candidates.begin(), candidates.end(), [](auto& a, auto& b) { return a.second > b.second; });
    for (auto& [t, benefit] : candidates) {
        if (validate_and_rebalance(graph, dev_state, features)) break;
        Device old_dev = t->device;
        Device new_dev = find_less_loaded_device(dev_state, features.at(t));
        if (new_dev != old_dev) {
            migrate_op(t, old_dev, new_dev, dev_state, features.at(t));
            std::println("  Migrated '{}' from {} to {}", t->name, device_to_string(old_dev), device_to_string(new_dev));
        }
    }
}

Device GraphScheduler::find_less_loaded_device(const DeviceState& dev_state, const OpFeature& op) {
    Device best = Device::CPU;
    double min_load = 1e18;
    
    for (const auto& [dev, info] : dev_state.info) {
        if (dev_state.available_memory(dev) < required_memory(op)) continue;
        
        double load = static_cast<double>(dev_state.stats.at(dev).assigned_flops) / 
                        info.compute_power;
        if (load < min_load) {
            min_load = load;
            best = dev;
        }
    }
    return best;
}
void GraphScheduler::migrate_op(
    Tensor* t, Device old_dev,
    Device new_dev,
    DeviceState& dev_state, OpFeature& op)
{
    size_t old_mem = dev_state.stats[old_dev].used_memory;
    t->device = new_dev;
    dev_state.stats[old_dev].used_memory -= required_memory(op);
    dev_state.stats[old_dev].assigned_flops -= op.flops;
    dev_state.stats[new_dev].used_memory += required_memory(op);
    dev_state.stats[new_dev].assigned_flops += op.flops;
    // 权重跟随
    if (op.weight_bytes > 0) {
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            Tensor* src = t->src[i];
            if (src && src->type == TensorType::TENSOR_TYPE_WEIGHT && 
                src->device == old_dev) {
                src->device = new_dev;
                dev_state.stats[old_dev].used_memory -= src->bytes();
                dev_state.stats[new_dev].used_memory += src->bytes();
            }
        }
    }
}

void GraphScheduler::allocate_memory(ComputeGraph& graph) {
    for (Tensor* t : graph.get_all_tensors()) {
        if (t->data != nullptr) continue;  // 已分配（如拷贝代理）
        
        auto* pool = get_pool(t->device);
        if (!pool) {
            std::println("Warning: No memory pool for {}", device_to_string(t->device));
            t->device = Device::CPU;
            pool = get_pool(Device::CPU);
        }
        
        t->offset = pool->allocate(t->bytes());
        t->data = pool->get_ptr(t->offset);
        t->backend = pool;
    }
}

size_t GraphScheduler::required_memory(const OpFeature& op) const {
    return op.weight_bytes + op.bytes_write + 
            static_cast<size_t>((op.bytes_read + op.bytes_write) * 0.2);
}
int64_t GraphScheduler::estimate_flops(Tensor* t) const {
    switch (t->op_type) {
        case OperationType::OP_TYPE_MAT_MUL:
        case OperationType::OP_TYPE_LINEAR:
            return t->num_elements() * 2;
        case OperationType::OP_TYPE_CONV2D:
            return t->num_elements() * 9 * 2;
        case OperationType::OP_TYPE_FLASH_ATTN:
        case OperationType::OP_TYPE_SDPA:
            return t->num_elements() * 4;
        case OperationType::OP_TYPE_RMS_NORM:
        case OperationType::OP_TYPE_ADD:
        case OperationType::OP_TYPE_MUL:
            return t->num_elements();
        default:
            return t->num_elements();
    }
}
void GraphScheduler::print_statistics(const DeviceState& dev_state) {
    std::println("\n=== Device Statistics ===");
    for (const auto& [dev, info] : dev_state.info) {
        const auto& stats = dev_state.stats.at(dev);
        auto* pool = pools_.at(dev).get();
        std::println(
            "{}: {} ops, {} TFLOPs, Memory: {} / {} GB ({:.1f}%)",
            device_to_string(dev),
            stats.assigned_ops,
            stats.assigned_flops / 1e12,
            pool->used_bytes() / 1024 / 1024 / 1024,
            pool->capacity_bytes() / 1024 / 1024 / 1024,
            pool->utilization()
        );
    }
}