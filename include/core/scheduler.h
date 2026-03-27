#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <queue>
#include <print>
#include <format>
#include "graph.hpp"
#include "tensor.hpp"
#include "backend/backend.h"
#include "memory_manager.h"
#include "utils/utils.hpp"

// 调度器类
// 
class GraphScheduler {
private:
    std::vector<BackendInfo> m_devices;
    std::unique_ptr<MemoryManager> m_memory_manager;
    
    int64_t estimate_weights_consumption(const ComputeGraph& graph){
        // 这个好算，直接统计TensorType为TENSOR_TYPE_WEIGHT的节点就行
    }
    int64_t estimate_avtivate_consumption(const ComputeGraph& graph){
        // 只能估计一个用量上限
    }
    int64_t estimate_kv_cache_consumption(const ComputeGraph& graph){
        // 认为是无上限，只返回一个合理的初始值。背后是用PageAttention方式管理的
    }
public:
    ~GraphScheduler();
    GraphScheduler(){};
    GraphScheduler(const GraphScheduler&) = delete;
    GraphScheduler& operator=(const GraphScheduler&) = delete;

    // 利用读取到的硬件信息，对计算节点指派执行设备

    // 综合这个计算图，判断权重参数存储消耗，临时节点存储消耗，kv-cache存储消耗。同时还要在对应设备上分配足够的内存/显存
    void graph_backend_assignment(const ComputeGraph& graph,const std::vector<BackendInfo>& backends){
        auto weight_bytes = this->estimate_weights_consumption(graph);
        auto activate_bytes = this->estimate_avtivate_consumption(graph);
        // auto kv_bytes = this->estimate_kv_cache_consumption(graph);
        
        
    }
};