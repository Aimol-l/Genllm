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
    std::vector<Tensor*> all_tensors_;      // 所有相关张量
    std::vector<Tensor*> execution_order_;  // 拓扑序（仅计算节点）
public:
    // 多输出逆向构建（只负责收集节点 + 拓扑排序）
    void build_from_outputs(std::initializer_list<Tensor*> outputs) {
        for (Tensor* out : outputs) {
            if (out) this->collect_tensors(out);
        }
        this->topological_sort();  // 生成纯依赖序，不含设备信息
    }
    // 获取拓扑序（供调度器使用）
    const std::vector<Tensor*>& get_execution_order() const {
        return execution_order_;
    }
    // 获取所有张量（供权重预加载使用）
    const std::vector<Tensor*>& get_all_tensors() const {
        return all_tensors_;
    }
    void clear(){
        all_tensors_.clear();
        execution_order_.clear();
    }
private:
    // 逆向收集所有 Tensor（BFS + 去重）
    void collect_tensors(Tensor* output) {
        std::queue<Tensor*> q;
        std::unordered_set<Tensor*> visited;
        q.push(output);
        visited.insert(output);
        while (!q.empty()) {
            Tensor* t = q.front(); q.pop();
            all_tensors_.push_back(t);
            for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
                Tensor* src = t->src[i];
                if (src && !visited.count(src)) {
                    visited.insert(src);
                    q.push(src);
                }
            }
        }
    }
    // Kahn 算法：对"计算产生的张量"排序
    void topological_sort() {
        // 入度 = 有多少个"计算张量"依赖当前张量作为输入
        std::unordered_map<Tensor*, int> in_degree;
        std::unordered_map<Tensor*, std::vector<Tensor*>> dependents;
        for (Tensor* t : all_tensors_) {
            if (!is_computed_tensor(t)) continue;
            
            in_degree[t] = 0;
            for (int i = 0; i < TENSOR_MAX_SRC; ++i) {
                Tensor* src = t->src[i];
                if (src && is_computed_tensor(src)) {
                    dependents[src].push_back(t);
                    in_degree[t]++;
                }
            }
        }
        std::queue<Tensor*> q;
        for (auto& [t, deg] : in_degree) {
            if (deg == 0) q.push(t);
        }
        while (!q.empty()) {
            Tensor* cur = q.front(); q.pop();
            execution_order_.push_back(cur);
            
            for (Tensor* dep : dependents[cur]) {
                if (--in_degree[dep] == 0) {
                    q.push(dep);
                }
            }
        }
    }
    bool is_computed_tensor(Tensor* t) {
        return t && t->op_type != OperationType::OP_TYPE_NONE 
            && t->type != TensorType::TENSOR_TYPE_WEIGHT 
            && t->type != TensorType::TENSOR_TYPE_INPUT;
    }
};