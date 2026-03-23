// graph.h - 计算图定义
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
#include "tensor.hpp"
#include "gguf_parser.h"
#include "utils/utils.hpp"

// 计算图类
// graph.h
class ComputeGraph {
public:
    // 添加节点 (图拥有所有权)
    void add_tensor(Tensor* t) {
        if (!t) return;
        tensors_.push_back(t);
        tensor_index_[t->name] = tensors_.size() - 1;  // 名称索引
    }
    
    // 标记输入/输出
    void set_input(const std::string& name, Tensor* t) {
        inputs_[name] = t;
    }
    void set_output(const std::string& name, Tensor* t) {
        outputs_[name] = t;
    }
    
    // 构建拓扑序
    void build_exec_order();
    
    // 访问器
    Tensor* get_tensor(const std::string& name) {
        auto it = tensor_index_.find(name);
        return (it != tensor_index_.end()) ? tensors_[it->second] : nullptr;
    }
    Tensor* get_output(const std::string& name) {
        auto it = outputs_.find(name);
        return (it != outputs_.end()) ? it->second : nullptr;
    }
    
    // 清理 (调用者负责 delete 图本身)
    void clear() {
        for (auto* t : tensors_) delete t;
        tensors_.clear();
        tensor_index_.clear();
        inputs_.clear();
        outputs_.clear();
        exec_order_.clear();
    }
    
    ~ComputeGraph() { clear(); }  // 析构时释放所有 Tensor
    
private:
    std::vector<Tensor*> tensors_;                    // 拥有所有权
    std::unordered_map<std::string, size_t> tensor_index_;  // 名称→索引
    std::unordered_map<std::string, Tensor*> inputs_;       // 输入节点
    std::unordered_map<std::string, Tensor*> outputs_;      // 输出节点
    std::vector<size_t> exec_order_;                  // 拓扑序索引
};
