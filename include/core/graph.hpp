#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <string>
#include <cstdint>
#include <print>
#include <format>
#include <fstream>
#include <queue>

#include "tensor.hpp"
#include "gguf_parser.h"
#include "utils/utils.hpp"

class ComputeGraph {
private: 
    std::vector<Tensor*> all_tensors_;          // 所有张量
    std::vector<Tensor*> execution_order_;      // 拓扑排序后的计算张量
    std::vector<Tensor*> external_outputs_;     // 外部可见的输出张量
public:
    void clear(){
        all_tensors_.clear();
        execution_order_.clear();
        external_outputs_.clear();
    }
    // 从输出张量逆向构建计算图
    void build_from_outputs(std::initializer_list<Tensor*> outputs) {
        for (Tensor* out : outputs) {
            if (out) {
                external_outputs_.push_back(out);
                collect_tensors(out);
            }
        }
        topological_sort();
        std::println("ComputeGraph built: {} tensors, {} compute ops", all_tensors_.size(), execution_order_.size());
    }
    // 单输出兼容接口
    void build_from_output(Tensor* output) {
        build_from_outputs({output});
    }
    
    const std::vector<Tensor*>& get_all_tensors() const { return all_tensors_; }
    const std::vector<Tensor*>& get_execution_order() const { return execution_order_; }
    const std::vector<Tensor*>& get_external_outputs() const { return external_outputs_; }
    
    // 替换输出指针（用于 D2H 拷贝后）
    void replace_output(Tensor* old_out, Tensor* new_out) {
        for (auto& out : external_outputs_) {
            if (out == old_out) {
                out = new_out;
                break;
            }
        }
    }
    
    // 获取总层数（用于调度器）
    int get_total_layers() const {
        int max_layer = -1;
        for (Tensor* t : all_tensors_) {
            int layer = parse_layer_from_name(t->name);
            if (layer >= 0) max_layer = std::max(max_layer, layer);
        }
        return max_layer + 1;
    }

private:
    // 逆向收集所有相关张量（BFS + 去重）
    void collect_tensors(Tensor* output) {
        std::queue<Tensor*> q;
        std::unordered_set<Tensor*> visited;
        
        q.push(output);
        visited.insert(output);
        
        while (!q.empty()) {
            Tensor* t = q.front(); q.pop();
            all_tensors_.push_back(t);
            
            // 遍历输入源
            for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
                Tensor* src = t->src[i];
                if (src && !visited.count(src)) {
                    visited.insert(src);
                    q.push(src);
                }
            }
        }
    }
    // Kahn 算法拓扑排序
    void topological_sort() {
        // 构建依赖图：计算张量之间的依赖
        std::unordered_map<Tensor*, int> in_degree;
        std::unordered_map<Tensor*, std::vector<Tensor*>> dependents;
        
        for (Tensor* t : all_tensors_) {
            if (!t->is_computed()) continue;
            
            in_degree[t] = 0;
            for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
                Tensor* src = t->src[i];
                if (src && src->is_computed()) {
                    dependents[src].push_back(t);
                    in_degree[t]++;
                }
            }
        }
        
        // 入度为 0 的节点入队
        std::queue<Tensor*> q;
        for (auto& [t, deg] : in_degree) {
            if (deg == 0) q.push(t);
        }
        
        // BFS 排序
        while (!q.empty()) {
            Tensor* cur = q.front(); q.pop();
            execution_order_.push_back(cur);
            
            for (Tensor* dep : dependents[cur]) {
                if (--in_degree[dep] == 0) {
                    q.push(dep);
                }
            }
        }
        
        // 检查环
        if (execution_order_.size() != in_degree.size()) {
            throw std::runtime_error("ComputeGraph has cycle!");
        }
    }
    // 从张量名解析层号（如 "blk.5.attn_q.weight" → 5）
    int parse_layer_from_name(const std::string& name) const {
        auto pos = name.find("blk.");
        if (pos != std::string::npos) {
            auto start = pos + 4;
            auto end = name.find('.', start);
            if (end != std::string::npos) {
                try {
                    return std::stoi(name.substr(start, end - start));
                } catch (...) {}
            }
        }
        return -1;
    }

};
