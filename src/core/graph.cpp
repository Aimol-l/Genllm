#include <format>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "core/graph.h"
#include "utils/tools.hpp"
// ========== private ==========

void ComputeGraph::reverse_bfs_collect(const std::vector<Tensor*>& seeds) {
    std::unordered_set<Tensor*> visited;
    std::queue<Tensor*> q;
    for (Tensor* t : seeds) {
        if (t && visited.insert(t).second) q.push(t);
    }
    while (!q.empty()) {
        Tensor* cur = q.front(); q.pop();
        all_tensors_.push_back(cur);
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            if (cur->src[i] && visited.insert(cur->src[i]).second)
                q.push(cur->src[i]);
        }
    }
}

void ComputeGraph::topological_sort() {
    std::unordered_map<Tensor*, int> in_degree;
    std::unordered_multimap<Tensor*, Tensor*> fwd;

    for (Tensor* t : all_tensors_) in_degree[t] = 0;
    for (Tensor* t : all_tensors_) {
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            Tensor* s = t->src[i];
            if (s && in_degree.count(s)) {
                ++in_degree[t];
                fwd.emplace(s, t);
            }
        }
    }

    // 按层分组：同一 level 内的节点 in_degree 同时归零，互不依赖可并行
    execution_levels_.clear();
    execution_order_.clear();
    layer_groups_.clear();
    max_layer_ = -1;
    size_t count = 0;

    while (count < all_tensors_.size()) {
        // 收集当前所有 in_degree==0 的节点作为同一 level
        std::vector<Tensor*> level;
        for (Tensor* t : all_tensors_) {
            if (in_degree[t] == 0) {
                level.push_back(t);
            }
        }
        if (level.empty()) break; // 没有可执行的节点 → 存在环

        // 层内按 layer_id 排序保持确定性
        std::sort(level.begin(), level.end(), [](Tensor* a, Tensor* b) {
            return a->layer_id < b->layer_id;
        });

        // 将该 level 的节点移除，更新依赖
        for (Tensor* t : level) {
            in_degree[t] = -1; // 标记已处理
            if (t->is_computed()) {
                execution_order_.push_back(t);
                layer_groups_[t->layer_id].push_back(t);
            }
            if (t->layer_id > max_layer_) max_layer_ = t->layer_id;
            ++count;
            auto range = fwd.equal_range(t);
            for (auto it = range.first; it != range.second; ++it) {
                if (in_degree[it->second] >= 0)
                    --in_degree[it->second];
            }
        }
        execution_levels_.push_back(std::move(level));
    }

    if (count != all_tensors_.size()) {
        throw std::runtime_error(
            std::format("ComputeGraph: cycle detected, {}/{} resolved", count, all_tensors_.size()));
    }
}

std::string ComputeGraph::dot_id(const Tensor* t) {
    return std::format("L{}_{}", t->layer_id, t->name);
}

// ========== public ==========

ComputeGraph::ComputeGraph(ComputeGraph&& other) noexcept
    : all_tensors_(std::move(other.all_tensors_))
    , execution_order_(std::move(other.execution_order_))
    , execution_levels_(std::move(other.execution_levels_))
    , external_outputs_(std::move(other.external_outputs_))
    , layer_groups_(std::move(other.layer_groups_))
    , max_layer_(other.max_layer_)
{
    other.max_layer_ = -1;
}

ComputeGraph& ComputeGraph::operator=(ComputeGraph&& other) noexcept {
    if (this != &other) {
        all_tensors_ = std::move(other.all_tensors_);
        execution_order_ = std::move(other.execution_order_);
        execution_levels_ = std::move(other.execution_levels_);
        external_outputs_ = std::move(other.external_outputs_);
        layer_groups_ = std::move(other.layer_groups_);
        max_layer_ = other.max_layer_;
        other.max_layer_ = -1;
    }
    return *this;
}

void ComputeGraph::build_from_outputs(std::initializer_list<Tensor*> outputs) {
    this->clear();
    this->external_outputs_.assign(outputs);
    this->reverse_bfs_collect(external_outputs_);
    this->topological_sort();
}

void ComputeGraph::clear() {
    all_tensors_.clear();
    execution_order_.clear();
    execution_levels_.clear();
    external_outputs_.clear();
    layer_groups_.clear();
    max_layer_ = -1;
}

void ComputeGraph::rebuild_order() {
    execution_order_.clear();
    execution_levels_.clear();
    layer_groups_.clear();
    max_layer_ = -1;
    topological_sort();
}

void ComputeGraph::replace_output(Tensor* old_t, Tensor* new_t) {
    for (auto& out : external_outputs_) {
        if (out == old_t) out = new_t;
    }
}

void ComputeGraph::add_tensor(Tensor* t) {
    all_tensors_.push_back(t);
}

Tensor* ComputeGraph::insert_memcpy(Tensor* original, Device dst_dev) {
    auto* proxy = new Tensor();
    proxy->name = "memcpy_" + original->name;
    proxy->op_type = OperationType::OP_TYPE_MEMCPY;
    proxy->type = original->type;
    proxy->device = dst_dev;
    proxy->dtype = original->dtype;
    proxy->dims = original->dims;
    proxy->layer_id = original->layer_id;
    proxy->src[0] = original;
    add_tensor(proxy);
    return proxy;
}

void ComputeGraph::export_dot(const std::string& path) const {
    std::ofstream os(path);
    if (!os) throw std::runtime_error(std::format("Cannot open file: {}", path));

    os << "digraph ComputeGraph {\n"
       << "  rankdir=TB;\n"
       << "  node [shape=box, style=filled];\n\n";

    for (auto* t : all_tensors_) {
        const char* color = [t] {
            if (t->type == TensorType::TENSOR_TYPE_WEIGHT)    return "#FFD54F";
            if (t->type == TensorType::TENSOR_TYPE_INPUT)     return "#90CAF9";
            if (t->type == TensorType::TENSOR_TYPE_OUTPUT)    return "#ff0000";
            if (t->type == TensorType::TENSOR_TYPE_CACHE)     return "#4a91d8";
            return "#C8E6C9";
        }();

        std::string layer_prefix = (t->layer_id >= 0) ? std::format("[L{}] ", t->layer_id) : "[global] ";
        std::string label = layer_prefix + t->name;

        if (t->op_type != OperationType::OP_TYPE_NONE)
            label += "\\n" + operation_type_to_string(t->op_type);

        label += "\\n{" + device_to_string(t->device) + "}";

        os << std::format("  \"{}\" [fillcolor=\"{}\", label=\"{}\"];\n",dot_id(t), color, label);
    }
    os << "\n";
    for (auto* t : all_tensors_) {
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            if (!t->src[i])
                continue;
            const char* style = (t->type == TensorType::TENSOR_TYPE_VIEW) ? "style=dashed" : "";
            os << std::format("  \"{}\" -> \"{}\" [{}];\n",dot_id(t->src[i]), dot_id(t), style);
        }
    }

    os << "}\n";
}
