#include "core/scheduler.h"

void GraphScheduler::schedule(const std::vector<BackendInfo>& devices) {
    if (devices.empty())
        throw std::runtime_error("GraphScheduler: no devices provided");
    auto costs = this->estimate_layer_costs(this->graph_); 
    if (costs.empty()) {
        std::println("[Scheduler] No transformer layers found, nothing to schedule");
        return;
    }
    this->assignments_ = this->assign_layers(costs, devices);
    this->apply_assignment(this->graph_, this->assignments_);
    
    Device cpu = Device::CPU;
    for (const auto& d : devices) {
        if (d.device == Device::CPU) { 
            cpu = d.device;
            break; 
        }
    }
    this->assign_global_nodes(this->graph_, cpu);
    this->insert_copy_edges(this->graph_);
    this->create_memory_pools(this->graph_, devices);
    this->print_summary(costs, devices);
}
// ========== 1. 估算每层内存开销 ==========
std::vector<GraphScheduler::LayerCost> GraphScheduler::estimate_layer_costs(const ComputeGraph& graph) const {
    const auto& all = graph.get_all_tensors();
    const auto& groups = graph.get_layer_groups();
    std::vector<LayerCost> costs;
    for (const auto& [layer_id, tensors] : groups) {
        // if (layer_id < 0) 
            // continue;
        LayerCost lc;
        lc.layer_id = layer_id;
        lc.kv_cache_bytes = config_.kv_cache_per_layer;
        for (auto* t : tensors) {
            if (t->type == TensorType::TENSOR_TYPE_VIEW) 
                continue;
            lc.activation_bytes += t->bytes_at(config_.max_seq_len);
        }
        costs.push_back(lc);
    }
    std::unordered_map<int, size_t> weight_map;
    for (auto* t : all) {
        if (t->layer_id < 0) 
            continue;
        if (t->type != TensorType::TENSOR_TYPE_WEIGHT) 
            continue;
        weight_map[t->layer_id] += t->bytes();
    }
    for (auto& lc : costs) {
        lc.weight_bytes = weight_map[lc.layer_id];
    }
    std::sort(costs.begin(), costs.end(),[](const LayerCost& a, const LayerCost& b) { return a.layer_id < b.layer_id; });
    return costs;
}

// ========== 2. 分配连续层到设备 ==========
std::vector<GraphScheduler::LayerAssignment> GraphScheduler::assign_layers(
    const std::vector<LayerCost>& costs,
    const std::vector<BackendInfo>& devices
) const{
    std::vector<const BackendInfo*> compute_devs;
    const BackendInfo* cpu_dev = nullptr;
    for (const auto& d : devices) {
        if (d.device == Device::CPU) cpu_dev = &d;
        else compute_devs.push_back(&d);
    }
    std::sort(compute_devs.begin(), compute_devs.end(),
              [](const BackendInfo* a, const BackendInfo* b) {
                  return a->available_memory() > b->available_memory();
              });
    if (compute_devs.empty()) {
        if (!costs.empty())
            return {{costs.front().layer_id, costs.back().layer_id, Device::CPU, 0, 0, 0}};
        return {};
    }
    size_t total = 0;
    for (const auto& c : costs) 
        total += c.total();
    size_t n = compute_devs.size();
    size_t per_dev = total / n + (total % n != 0 ? 1 : 0);
    std::vector<LayerAssignment> result;
    size_t accumulated = 0;
    int dev_idx = 0;
    int range_start = costs.front().layer_id;
    size_t range_bytes = 0;
    size_t range_weight = 0;
    size_t range_act = 0;
    for (size_t i = 0; i < costs.size(); ++i) {
        const auto& cost = costs[i];
        size_t layer_bytes = cost.total();
        bool switch_dev = false;
        if (i > 0 && accumulated > 0) {
            size_t budget = static_cast<size_t>(
                compute_devs[dev_idx]->available_memory() * (1.0 - config_.memory_headroom));
            if (accumulated + layer_bytes > budget ||
                (accumulated >= per_dev && dev_idx + 1 < static_cast<int>(n))) {
                switch_dev = true;
            }
        }
        if (switch_dev && dev_idx + 1 < static_cast<int>(n)) {
            result.push_back({range_start, costs[i - 1].layer_id,
                              compute_devs[dev_idx]->device, range_bytes,
                              range_weight, range_act});
            ++dev_idx;
            range_start = cost.layer_id;
            accumulated = 0;
            range_bytes = 0;
            range_weight = 0;
            range_act = 0;
        }
        accumulated += layer_bytes;
        range_bytes += layer_bytes;
        range_weight += cost.weight_bytes;
        range_act += cost.activation_bytes;
    }
    result.push_back({range_start, costs.back().layer_id, compute_devs[dev_idx]->device,
                      range_bytes, range_weight, range_act});
    return result;
}

// ========== 3. 把分配结果写入 Tensor::device ==========
void GraphScheduler::apply_assignment(
    ComputeGraph& graph,
    const std::vector<LayerAssignment>& assignments) const
{
    std::unordered_map<int, Device> layer_dev;
    for (const auto& a : assignments) {
        for (int l = a.start_layer; l <= a.end_layer; ++l) {
            layer_dev[l] = a.device;
        }
    }

    for (auto* t : graph.get_all_tensors()) {
        if (t->layer_id < 0) continue;
        auto it = layer_dev.find(t->layer_id);
        if (it != layer_dev.end()) {
            t->device = it->second;
        }
    }
}

// ========== 4. 全局节点分配到 CPU ==========
void GraphScheduler::assign_global_nodes(ComputeGraph& graph, Device cpu) const {
    for (auto* t : graph.get_all_tensors()) {
        if (t->layer_id >= 0) continue;
        t->device = cpu;
    }
}

// ========== 5. 打印分配摘要 ==========
void GraphScheduler::print_summary(
    const std::vector<LayerCost>& costs,
    const std::vector<BackendInfo>& devices) const
{
    std::println("=== GraphScheduler Summary ===");
    std::println("  Per-layer cost estimate:");
    for (const auto& c : costs) {
        std::println("    L{:>3d}: weight={}  activation={}  kv_cache={}  total={}",
                     c.layer_id,
                     format_bytes(c.weight_bytes),
                     format_bytes(c.activation_bytes),
                     format_bytes(c.kv_cache_bytes),
                     format_bytes(c.total()));
    }
    std::println("  Device assignments:");
    for (const auto& a : this->assignments_) {
        int n_layers = a.end_layer - a.start_layer + 1;
        std::println("    {} : L{} ~ L{} ({} layers, {})",
                     device_to_string(a.device),
                     a.start_layer, a.end_layer, n_layers,
                     format_bytes(a.total_bytes));
    }
    size_t total_weight = 0, total_act = 0;
    for (const auto& c : costs) {
        total_weight += c.weight_bytes;
        total_act += c.activation_bytes;
    }
    std::println("  Total: {} weights, {} activations(estimate)",format_bytes(total_weight), format_bytes(total_act));
    std::println("================================================");
}

// ========== 6. 插入跨设备拷贝边 ==========
void GraphScheduler::insert_copy_edges(ComputeGraph& graph) const {
    struct Pair {
        Tensor* src;
        Device dst;
        bool operator==(const Pair& o) const { return src == o.src && dst == o.dst; }
    };
    struct PairHash {
        size_t operator()(const Pair& p) const {
            return std::hash<Tensor*>()(p.src) ^ (static_cast<size_t>(p.dst) << 32);
        }
    };

    std::unordered_map<Pair, Tensor*, PairHash> cache;
    std::vector<std::pair<Tensor*, Tensor*>> replacements;
    int deduped = 0;

    // 遍历所有张量的输入，检查是否跨设备，如果是则插入一个 memcpy 节点，并更新输入指向
    for (auto* t : graph.get_all_tensors()) {
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            Tensor* src = t->src[i];
            if (!src || src->device == t->device) 
                continue;
            Pair key{src, t->device};
            auto it = cache.find(key);
            if (it != cache.end()) {
                t->src[i] = it->second;
                ++deduped;
                continue;
            }
            Tensor* proxy = graph.insert_memcpy(src, t->device);
            cache[key] = proxy;
            t->src[i] = proxy;
            ++deduped;
            std::println("  [copy] {} ({}) -> {} ({})",src->name, device_to_string(src->device),proxy->name, device_to_string(t->device));
        }
    }
    for(auto* t:graph.get_external_outputs()){
        if(t->device != Device::CPU){
            Pair key{t, Device::CPU};
            auto it = cache.find(key);
            if (it != cache.end()) {
                graph.replace_output(t, it->second);
                ++deduped;
                continue;
            }
            Tensor* proxy = graph.insert_memcpy(t, Device::CPU);
            cache[key] = proxy;
            graph.replace_output(t, proxy);
            ++deduped;
            std::println("  [copy] {} ({}) -> {} ({})",
                         t->name, device_to_string(t->device),
                         proxy->name, device_to_string(Device::CPU));
        }
    }
    if (deduped > 0) {
        graph.rebuild_order();
        std::println("Inserted {} copy nodes ({} deduplicated refs)",cache.size(), deduped);
    }
}

void GraphScheduler::create_memory_pools(const ComputeGraph& graph,const std::vector<BackendInfo>& devices) {
    std::unordered_map<Device, size_t> dev_id_map;
    for (const auto& d : devices) {
        dev_id_map[d.device] = d.id;
    }
    struct DeviceMemUsage {
        size_t weight_bytes = 0;
        size_t activation_bytes = 0;
    };
    std::unordered_map<Device, DeviceMemUsage> usage;
    for (auto* t : graph.get_all_tensors()) {
        if (t->type == TensorType::TENSOR_TYPE_VIEW)
            continue;
        if (t->type == TensorType::TENSOR_TYPE_WEIGHT) {
            usage[t->device].weight_bytes += t->bytes();
        } else {
            usage[t->device].activation_bytes += t->bytes_at(config_.max_seq_len);
        }
    }

    for (auto& [dev, u] : usage) {
        size_t dev_id = dev_id_map[dev];
        size_t act_cap = u.activation_bytes * config_.activation_pool_factor;
        if (act_cap < 64ULL << 20) 
            act_cap = 64ULL << 20;
        this->mmanager_->get_or_create(dev, dev_id, u.weight_bytes, act_cap, 0);
        std::println("{}:{}  weight_pool={}  activation_pool={}",device_to_string(dev), dev_id,format_bytes(u.weight_bytes), format_bytes(act_cap));
    }
}