#include <cstddef>
#include <print>

#include "core/scheduler.h"
#include "core/page_attention.h"

void GraphScheduler::schedule(const std::vector<BackendInfo>& devices) {
    if (devices.empty())
        throw std::runtime_error("GraphScheduler: no devices provided");

    // 1. 估算每层内存开销（激活缓存 + 权重 + KV cache）
    std::vector<LayerCost> costs = this->estimate_layer_costs(this->graph_); 
    if (costs.empty()) {
        std::println("[Scheduler] No transformer layers found, nothing to schedule");
        return;
    }
    this->assignments_ = this->assign_layers(costs, devices); // 2. 分配连续层到设备

    this->apply_assignment(this->graph_, this->assignments_); // 实际进行设备分配
    
    Device cpu = Device::CPU;
    for (const auto& d : devices) {
        if (d.device == Device::CPU) { 
            cpu = d.device;
            break; 
        }
    }
    this->assign_global_nodes(this->graph_, cpu); // 4. 添加全局节点，如rope_sin / cos
    this->insert_copy_edges(this->graph_);         // 为跨设备情况添加拷贝节点
    this->create_memory_pools(this->graph_, devices); // 创建内存池
    this->initialize_kv_cache();                       // 初始化 KV cache 的 page table
    this->print_summary(costs, devices);
}
// ========== 1. 估算每层内存开销 ==========
std::vector<GraphScheduler::LayerCost> GraphScheduler::estimate_layer_costs(const ComputeGraph& graph) const {
    const auto& all = graph.get_all_tensors();
    const auto& groups = graph.get_layer_groups();
    std::vector<LayerCost> costs;
    for (const auto& [layer_id, tensors] : groups) {
        LayerCost lc;
        lc.layer_id = layer_id;
        for (auto* t : tensors) {
            if (t->type == TensorType::TENSOR_TYPE_VIEW) 
                continue;
            lc.activation_bytes += t->bytes_at(config_.max_seq_len);
            
        }
        costs.push_back(lc);
    }
    std::unordered_map<int, size_t> weight_map;
    for (auto* t : all) {
        // if (t->layer_id < 0) 
            // continue;
        if (t->type != TensorType::TENSOR_TYPE_WEIGHT) 
            continue;
        weight_map[t->layer_id] += t->bytes();
    }
    for (auto& lc : costs) {
        lc.weight_bytes = weight_map[lc.layer_id];
        lc.activation_bytes = static_cast<size_t>(lc.activation_bytes * config_.activation_pool_factor);
    }
    // KV cache: 每层找 SDPA 张量，用 K weight 的 dims 推算 n_kv_heads 和 head_dim
    {
        int32_t max_blocks = (config_.max_seq_len + PAGE_BLOCK_SIZE - 1) / PAGE_BLOCK_SIZE;
        for (const auto& [layer_id, tensors] : groups) {
            for (auto* t : tensors) {
                if (t->op_type != OperationType::OP_TYPE_SDPA || !t->src[1]) continue;
                int32_t n_kv_heads = static_cast<int32_t>(t->src[1]->dims[1]);
                int32_t head_dim   = static_cast<int32_t>(t->dims[3]);
                size_t block_bytes = static_cast<size_t>(PAGE_BLOCK_SIZE) * n_kv_heads * head_dim * data_type_size(t->dtype);
                size_t kv_bytes    = 2 * static_cast<size_t>(max_blocks) * block_bytes;
                for (auto& lc : costs) {
                    if (lc.layer_id == layer_id) {
                        lc.kv_cache_bytes += static_cast<size_t>(kv_bytes * config_.kv_cache_pool_factor);
                        break;
                    }
                }
            }
        }
    }
    std::sort(costs.begin(), costs.end(),[](const LayerCost& a, const LayerCost& b) { return a.layer_id < b.layer_id; });
    return costs;
}

// 返回值：每段对应一个 LayerAssignment（start_layer, end_layer, device, ...）。
// 策略：按层顺序依次填充设备，每层实际占用的内存为：
//   Σ(weight_i + kv_cache_i) + max(activation_i)   （激活层间复用，只保留最大值）
// 若该段尾部需要跨设备拷贝，额外预留一份 max(activation) 作为拷贝缓冲区。
// 当当前设备放不下下一层时，切到下一个设备。所有计算设备用完后回退 CPU。
// 若 CPU 也放不下剩余层，抛出异常。
// 注：copy 节点的精确开销由 create_memory_pools 在实际插入后计算，从 kv_cache 池扣减。
std::vector<GraphScheduler::LayerAssignment> GraphScheduler::assign_layers(
    const std::vector<LayerCost>& costs,
    const std::vector<BackendInfo>& devices
) const{
    if (costs.empty()) return {};

    std::vector<const BackendInfo*> dev_queue;
    for (const auto& d : devices) {
        if (d.device != Device::CPU)
            dev_queue.push_back(&d);
    }
    std::sort(dev_queue.begin(), dev_queue.end(),
              [](const BackendInfo* a, const BackendInfo* b) {
                  return a->available_memory() > b->available_memory();
              });
    // CPU 作为最后兜底
    const BackendInfo* cpu_dev = nullptr;
    for (const auto& d : devices) {
        if (d.device == Device::CPU) { cpu_dev = &d; break; }
    }
    std::vector<LayerAssignment> result;
    int dev_idx   = 0;
    int layer_idx = 0;
    int n_layers  = static_cast<int>(costs.size());

    while (layer_idx < n_layers) {
        const BackendInfo* cur_dev = (dev_idx < static_cast<int>(dev_queue.size()))
                                    ? dev_queue[dev_idx] : cpu_dev;
        if (!cur_dev)
            throw std::runtime_error("assign_layers: 无可用设备，无法容纳剩余层");

        size_t budget = static_cast<size_t>(cur_dev->available_memory() * (1.0f - config_.memory_headroom));

        int  seg_start = costs[layer_idx].layer_id;
        int  seg_end   = seg_start;
        size_t sum_weight = 0;
        size_t sum_kv     = 0;
        size_t max_act    = 0;
        bool  seg_empty   = true;

        while (layer_idx < n_layers) {
            const auto& c = costs[layer_idx];
            size_t cur_act = std::max(max_act, c.activation_bytes);
            size_t need = sum_weight + c.weight_bytes
                        + sum_kv + c.kv_cache_bytes
                        + cur_act;

            if (need > budget && !seg_empty) break;
            sum_weight += c.weight_bytes;
            sum_kv     += c.kv_cache_bytes;
            max_act     = cur_act;
            seg_end = c.layer_id;
            seg_empty = false;
            ++layer_idx;
        }
        if (seg_empty) {
            size_t min_need = costs[layer_idx].weight_bytes + costs[layer_idx].kv_cache_bytes;
            throw std::runtime_error(std::format(
                "assign_layers: 设备 {}:{} 无法容纳第 {} 层 (需要至少 {}, 可用 {})",
                device_to_string(cur_dev->device), cur_dev->id,
                costs[layer_idx].layer_id,
                format_bytes(min_need), format_bytes(budget)));
        }
        result.push_back({seg_start, seg_end, cur_dev->device, cur_dev->id,sum_kv, sum_weight, max_act});
        ++dev_idx;
    }
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
        if (t->layer_id >= 0) 
            continue;
        t->device = cpu;
    }
}

// ========== 5. 打印分配摘要 ==========
void GraphScheduler::print_summary(
    const std::vector<LayerCost>& costs,
    const std::vector<BackendInfo>& devices) const
{
    std::println("=============GraphScheduler Summary=====================");
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
        std::println("    {} : L:{} ~ L:{} ({} layers, exc{},share{})",
                     device_to_string(a.device),
                     a.start_layer,
                     a.end_layer,
                     n_layers,
                     format_bytes(a.weight_bytes+a.kv_cache_bytes),
                     format_bytes(a.kv_cache_bytes)
                    );
    }
    size_t total_weight = 0, max_act = 0,total_kv=0;
    for (const auto& c : costs) {
        total_weight += c.weight_bytes;
        total_kv     += c.kv_cache_bytes;
        if (c.activation_bytes > max_act) max_act = c.activation_bytes;
    }
    std::println("  Total: {} weights, {} activations(max layer), {} kv-caches",format_bytes(total_weight), format_bytes(max_act),format_bytes(total_kv));
    std::println("========================================================");
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

    // 第一趟：只收集跨设备边，不改图（避免遍历时 push_back 导致迭代器失效）
    struct Edge { Tensor* src; Device dst_dev; int src_idx; Tensor* consumer; };
    std::vector<Edge> pending;
    for (auto* t : graph.get_all_tensors()) {
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            Tensor* src = t->src[i];
            if (!src || src->device == t->device) continue;
            pending.push_back({src, t->device, i, t});
        }
    }
    // 外部输出也需要可能的回拷
    for (auto* t : graph.get_external_outputs()) {
        if (t->device != Device::CPU) {
            pending.push_back({t, Device::CPU, -1, nullptr});
        }
    }

    // 第二趟：插入 memcpy 节点并更新引用
    std::unordered_map<Pair, Tensor*, PairHash> cache;
    int deduped = 0;
    for (auto& e : pending) {
        Pair key{e.src, e.dst_dev};
        auto it = cache.find(key);
        if (it != cache.end()) {
            if (e.consumer)
                e.consumer->src[e.src_idx] = it->second;
            else
                graph.replace_output(e.src, it->second);
            ++deduped;
            continue;
        }
        Tensor* proxy = graph.insert_memcpy(e.src, e.dst_dev);
        cache[key] = proxy;
        if (e.consumer)
            e.consumer->src[e.src_idx] = proxy;
        else
            graph.replace_output(e.src, proxy);
        ++deduped;
        std::println("  [copy] {} ({}) -> {} ({})",
                     e.src->name, device_to_string(e.src->device),
                     proxy->name, device_to_string(e.dst_dev));
    }

    if (deduped > 0) {
        graph.rebuild_order();
        std::println("Inserted {} copy nodes ({} deduplicated refs)", cache.size(), deduped);
    }
}

// 这里会再次统计需要使用的权重池、激活池使用大小
// 然后创建权重池、激活池
// 激活池大小：取每层激活的最大值（层间复用内存，不累加）
void GraphScheduler::create_memory_pools(const ComputeGraph& graph,const std::vector<BackendInfo>& devices) {
    std::unordered_map<Device, size_t> dev_id_map;
    std::unordered_map<Device, size_t> dev_budget;
    for (const auto& d : devices) {
        dev_id_map[d.device] = d.id;
        dev_budget[d.device] = static_cast<size_t>(d.available_memory() * (1.0 - config_.memory_headroom));
    }
    struct DeviceMemUsage {
        size_t weight_bytes     = 0;
        size_t activation_bytes = 0;
        size_t kv_cache_bytes   = 0;
        size_t copy_bytes       = 0;
    };
    std::unordered_map<Device, DeviceMemUsage> usage;

    // 权重池：累加所有权重
    for (auto* t : graph.get_all_tensors()) {
        if (t->type == TensorType::TENSOR_TYPE_VIEW) continue;
        if (t->type == TensorType::TENSOR_TYPE_WEIGHT) {
            usage[t->device].weight_bytes += t->bytes();
        }
    }

    // 激活池：按层统计，取最大单层激活量（层间复用，不累加）
    // 同时统计 copy 节点开销（insert_copy_edges 已执行完毕，图中已有 memcpy 节点）
    std::unordered_map<int, size_t> layer_act;
    for (auto* t : graph.get_all_tensors()) {
        if (t->type == TensorType::TENSOR_TYPE_VIEW) continue;
        if (t->type == TensorType::TENSOR_TYPE_WEIGHT) continue;
        if (t->op_type == OperationType::OP_TYPE_MEMCPY) {
            usage[t->device].copy_bytes += t->bytes_at(config_.max_seq_len);
            continue;
        }
        layer_act[t->layer_id] += t->bytes_at(config_.max_seq_len);
    }
    for (auto& [dev, u] : usage) {
        size_t max_act = 0;
        for (auto* t : graph.get_all_tensors()) {
            if (t->type == TensorType::TENSOR_TYPE_VIEW || t->type == TensorType::TENSOR_TYPE_WEIGHT) continue;
            if (t->op_type == OperationType::OP_TYPE_MEMCPY) continue;
            if (t->device != dev) continue;

            size_t la = layer_act[t->layer_id];
            if (la > max_act)
                max_act = la;
        }
        u.activation_bytes = static_cast<size_t>(max_act * config_.activation_pool_factor);

        // KV cache 池：累加该设备上所有层的 paged cache
        int32_t max_blocks = static_cast<int32_t>((config_.max_seq_len + PAGE_BLOCK_SIZE - 1) / PAGE_BLOCK_SIZE);
        for (auto* t : graph.get_all_tensors()) {
            if (t->op_type != OperationType::OP_TYPE_SDPA || !t->src[1]) continue;
            if (t->device != dev) continue;
            int32_t n_kv_heads = static_cast<int32_t>(t->src[1]->dims[1]);
            int32_t head_dim = static_cast<int32_t>(t->dims[3]);
            size_t block_bytes = static_cast<size_t>(PAGE_BLOCK_SIZE) * n_kv_heads * head_dim * data_type_size(t->dtype);
            u.kv_cache_bytes += static_cast<size_t>(2 * static_cast<size_t>(max_blocks) * block_bytes * config_.kv_cache_pool_factor);
        }
    }

    for (auto& [dev, u] : usage) {
        size_t dev_id = dev_id_map[dev];
        size_t act_cap = u.activation_bytes;
        size_t kv_cap = u.kv_cache_bytes;
        if (act_cap < 64ULL << 20)  act_cap = 64ULL << 20;

        // copy 节点额外占用显存，从 kv_cache 池中扣减
        if (u.copy_bytes > 0) {
            size_t budget = dev_budget[dev];
            size_t fixed = u.weight_bytes + act_cap + u.copy_bytes;
            if (fixed >= budget) {
                throw std::runtime_error(std::format(
                    "create_memory_pools: {}:{} 权重+激活+拷贝节点 ({}) 已超出可用预算 ({})",
                    device_to_string(dev), dev_id,
                    format_bytes(fixed), format_bytes(budget)));
            }
            size_t max_kv = budget - fixed;
            if (kv_cap > max_kv) {
                std::println("[Scheduler] {}:{} copy 节点占用 {}, kv_cache 池从 {} 缩减至 {}",
                    device_to_string(dev), dev_id,
                    format_bytes(u.copy_bytes), format_bytes(kv_cap), format_bytes(max_kv));
                kv_cap = max_kv;
            }
        }

        this->mmanager_->get_or_create(dev, dev_id, u.weight_bytes, act_cap, kv_cap);
    }
}

void GraphScheduler::initialize_kv_cache() {

    int32_t max_blocks = static_cast<int32_t>((config_.max_seq_len + PAGE_BLOCK_SIZE - 1) / PAGE_BLOCK_SIZE);

    // 按 assignment 的设备分组，每个 assignment 有正确的 dev_id
    for (const auto& assign : assignments_) {
        DevicePools* pools = mmanager_->get(assign.device, assign.dev_id);
        if (!pools || !pools->kv_cache) continue;

        PagedAttentionManager& pam = mmanager_->create_attention_manager(assign.device, assign.dev_id);
        
        for (int l = assign.start_layer; l <= assign.end_layer; ++l) {
            for (auto* t : graph_.get_all_tensors()) {
                if (t->op_type != OperationType::OP_TYPE_SDPA || !t->src[1]) continue;
                if (t->layer_id != l) continue;

                int32_t n_kv_heads = static_cast<int32_t>(t->src[1]->dims[1]);
                int32_t head_dim = static_cast<int32_t>(t->dims[3]);
                DataType dtype = t->dtype;
                pam.init_layer(t->layer_id, n_kv_heads, head_dim, dtype);
                pam.reserve_layer(t->layer_id, max_blocks);
            }
        }
    }
}