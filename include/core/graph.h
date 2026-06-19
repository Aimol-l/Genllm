#pragma once
#include <vector>
#include <map>
#include "tensor.hpp"

class ComputeGraph {
private:
    int max_layer_ = -1;

    std::vector<Tensor*> all_tensors_;
    std::vector<Tensor*> execution_order_;
    std::vector<Tensor*> external_outputs_;
    std::map<int, std::vector<Tensor*>> layer_groups_;
    std::vector<std::vector<Tensor*>> execution_levels_; // 按依赖层级分组，同层内节点可并行

    void topological_sort();
    static std::string dot_id(const Tensor* t);
    void reverse_bfs_collect(const std::vector<Tensor*>& seeds);
public:
    ComputeGraph() = default;
    ~ComputeGraph() {
        for (auto* t : all_tensors_) 
            delete t; // 只回收元数据，实际data由IMemoryResource管理
    }

    ComputeGraph(const ComputeGraph&) = delete;
    ComputeGraph& operator=(const ComputeGraph&) = delete;

    ComputeGraph(ComputeGraph&& other) noexcept;
    ComputeGraph& operator=(ComputeGraph&& other) noexcept;

    void build_from_outputs(std::initializer_list<Tensor*> outputs);
    void clear();
    void rebuild_order();

    const std::vector<Tensor*>& get_execution_order()  const { return execution_order_; }
    const std::vector<std::vector<Tensor*>>& get_execution_levels() const { return execution_levels_; }// 返回的都是计算节点
    const std::vector<Tensor*>& get_all_tensors()       const { return all_tensors_; }
    const std::vector<Tensor*>& get_external_outputs()  const { return external_outputs_; }
    const std::map<int, std::vector<Tensor*>>& get_layer_groups() const { return layer_groups_; }
    int get_max_layer() const { return max_layer_; }

    void replace_output(Tensor* old_t, Tensor* new_t);
    void add_tensor(Tensor* t);
    Tensor* insert_memcpy(Tensor* original, Device dst_dev);
    void export_dot(const std::string& path) const;
};
