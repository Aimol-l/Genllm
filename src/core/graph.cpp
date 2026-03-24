// graph.cpp - 计算图实现
#include "core/graph.h"
#include <queue>
#include <algorithm>
#include <format>
// graph.cpp
void ComputeGraph::collect_from_root(
    Tensor* root, 
    std::initializer_list<Tensor*> external_inputs
) {
    if (!root) throw std::runtime_error("collect_from_root: root is null");
    
    // 临时存储: 先收集原始指针的深拷贝
    std::unordered_map<const Tensor*, size_t> ptr_to_idx;
    std::vector<std::unique_ptr<Tensor>> temp_storage;
    
    // Step 1: DFS 收集所有节点 (去重)
    collect_dfs(root, ptr_to_idx, temp_storage);
    
    // Step 2: 标记外部输入
    for (Tensor* inp : external_inputs) {
        if (ptr_to_idx.count(inp)) {
            external_inputs_.insert(temp_storage[ptr_to_idx[inp]].get());
        }
    }
    
    // Step 3: 转移所有权到 tensors_
    tensors_ = std::move(temp_storage);
    
    // Step 4: 填充名称索引
    for (size_t i = 0; i < tensors_.size(); ++i) {
        if (!tensors_[i]->name.empty()) {
            name_to_idx_[tensors_[i]->name] = i;
        }
    }
    
    // Step 5: 修复 src 指针 (指向 tensors_ 内的副本)
    fix_src_pointers(ptr_to_idx);
    
    // Step 6: 拓扑排序
    build_topological_order();
    
    // Step 7: 初始化 use_count (用于运行时内存释放)
    use_count_.assign(tensors_.size(), 0);
    for (size_t i = 0; i < tensors_.size(); ++i) {
        const Tensor& t = *tensors_[i];
        for (int j = 0; j < TENSOR_MAX_SRC; ++j) {
            if (t.src[j]) {
                auto it = std::find_if(tensors_.begin(), tensors_.end(),
                    [src=t.src[j]](const auto& ptr) { return ptr.get() == src; });
                if (it != tensors_.end()) {
                    size_t src_idx = std::distance(tensors_.begin(), it);
                    use_count_[src_idx]++;
                }
            }
        }
    }
}

void ComputeGraph::collect_dfs(
    Tensor* src,
    std::unordered_map<const Tensor*, size_t>& ptr_to_idx,
    std::vector<std::unique_ptr<Tensor>>& temp_storage
) {
    if (!src || ptr_to_idx.count(src)) return;
    
    // 创建深拷贝: 拷贝元数据，但保留原始指针 (后续修复)
    size_t idx = temp_storage.size();
    ptr_to_idx[src] = idx;
    
    auto copy = std::make_unique<Tensor>(*src);  // 浅拷贝: data/src 指针值相同
    temp_storage.push_back(std::move(copy));
    
    // 递归收集源节点
    for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
        if (src->src[i]) {
            collect_dfs(src->src[i], ptr_to_idx, temp_storage);
        }
    }
}

void ComputeGraph::fix_src_pointers(
    const std::unordered_map<const Tensor*, size_t>& ptr_to_idx
) {
    // 重定向每个 Tensor 的 src[] 指向内部副本
    for (auto& t : tensors_) {
        for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
            if (t->src[i]) {
                auto it = ptr_to_idx.find(t->src[i]);
                if (it != ptr_to_idx.end()) {
                    t->src[i] = tensors_[it->second].get();  // ✅ 重定向
                }
            }
        }
    }
}

void ComputeGraph::build_topological_order() {
    // Kahn 算法
    std::vector<int> in_degree(tensors_.size(), 0);
    
    // 计算入度
    for (size_t i = 0; i < tensors_.size(); ++i) {
        for (int j = 0; j < TENSOR_MAX_SRC; ++j) {
            if (tensors_[i]->src[j]) {
                // 查找 src 在 tensors_ 中的索引
                auto it = std::find_if(tensors_.begin(), tensors_.end(),
                    [src=tensors_[i]->src[j]](const auto& ptr) { return ptr.get() == src; });
                if (it != tensors_.end()) {
                    in_degree[i]++;
                }
            }
        }
    }
    
    // BFS
    std::queue<size_t> q;
    for (size_t i = 0; i < tensors_.size(); ++i) {
        if (in_degree[i] == 0) {
            q.push(i);  // 叶子节点先执行
        }
    }
    
    while (!q.empty()) {
        size_t u = q.front(); q.pop();
        exec_order_.push_back(u);
        
        // 更新后继
        for (size_t v = 0; v < tensors_.size(); ++v) {
            for (int j = 0; j < TENSOR_MAX_SRC; ++j) {
                if (tensors_[v]->src[j] == tensors_[u].get()) {
                    if (--in_degree[v] == 0) {
                        q.push(v);
                    }
                }
            }
        }
    }
    
    if (exec_order_.size() != tensors_.size()) {
        throw std::runtime_error("ComputeGraph contains cycle!");
    }
}

void ComputeGraph::clear() {
    tensors_.clear();
    name_to_idx_.clear();
    inputs_.clear();
    outputs_.clear();
    external_inputs_.clear();
    exec_order_.clear();
    use_count_.clear();
}

void ComputeGraph::set_input(const std::string& name, Tensor* t) {
    // 在 tensors_ 中查找对应副本
    auto it = std::find_if(tensors_.begin(), tensors_.end(),
        [orig=t](const auto& ptr) { 
            return ptr->name == orig->name || ptr->data == orig->data; 
        });
    if (it != tensors_.end()) {
        inputs_[name] = it->get();
    }
}

void ComputeGraph::set_output(const std::string& name, Tensor* t) {
    auto it = std::find_if(tensors_.begin(), tensors_.end(),
        [orig=t](const auto& ptr) { 
            return ptr->name == orig->name || ptr->data == orig->data; 
        });
    if (it != tensors_.end()) {
        outputs_[name] = it->get();
    }
}

Tensor* ComputeGraph::get_tensor(const std::string& name) const {
    auto it = name_to_idx_.find(name);
    return (it != name_to_idx_.end()) ? tensors_[it->second].get() : nullptr;
}

Tensor* ComputeGraph::get_output(const std::string& name) const {
    auto it = outputs_.find(name);
    return (it != outputs_.end()) ? it->second : nullptr;
}