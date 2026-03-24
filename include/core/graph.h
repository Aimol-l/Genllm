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
    // ✅ 核心: 从输出节点反向遍历，自动收集所有依赖的 Tensor
    // 参数:
    //   root: 计算图的输出节点 (如 logits)
    //   external_inputs: 外部输入列表 (标记为"不自动释放")
    void collect_from_root(Tensor* root, 
                          std::initializer_list<Tensor*> external_inputs = {});
    
    // 访问器
    Tensor* get_tensor(const std::string& name) const;
    Tensor* get_output(const std::string& name) const;
    const std::vector<size_t>& exec_order() const { return exec_order_; }
    size_t tensor_count() const { return tensors_.size(); }
    size_t exec_order_size() const { return exec_order_.size(); }
    
    // 标记输入/输出
    void set_input(const std::string& name, Tensor* t);
    void set_output(const std::string& name, Tensor* t);
    
    // 清理
    void clear();
    
private:
    // 内部实现
    void collect_dfs(Tensor* src, 
                    std::unordered_map<const Tensor*, size_t>& ptr_to_idx,
                    std::vector<std::unique_ptr<Tensor>>& temp_storage);
    void fix_src_pointers(const std::unordered_map<const Tensor*, size_t>& ptr_to_idx);
    void build_topological_order();
    
    // 数据成员
    std::vector<std::unique_ptr<Tensor>> tensors_;                    // 拥有所有权
    std::unordered_map<std::string, size_t> name_to_idx_;             // 名称→索引
    std::unordered_map<std::string, Tensor*> inputs_;                 // 输入节点
    std::unordered_map<std::string, Tensor*> outputs_;                // 输出节点
    std::unordered_set<Tensor*> external_inputs_;                     // 不自动释放的节点
    std::vector<size_t> exec_order_;                                  // 拓扑序
    std::vector<int> use_count_;                                      // 运行时引用计数
};