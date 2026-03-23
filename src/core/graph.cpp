// graph.cpp - 计算图实现
#include "core/graph.h"
#include <queue>
#include <algorithm>
#include <format>

// 导出为 Graphviz dot 文件
void ComputeGraph::export_dot(const std::string& filename) const {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Failed to open file: {}", filename));
    }

    ofs << "digraph ComputationGraph {\n";
    ofs << "    rankdir=TB;\n";
    ofs << "    node [shape=box, style=rounded];\n\n";

    std::unordered_set<Tensor*> visited;

    // 导出所有叶子节点（参数）
    ofs << "    // Leaf nodes (parameters)\n";
    for (const auto* leaf : m_leafs) {
        if (visited.find(leaf) != visited.end()) continue;
        visited.insert(leaf);

        std::string node_id = std::format("tensor_0x{:x}", reinterpret_cast<uintptr_t>(leaf));
        std::string label = std::format(
            "{}\n{}\n{}",
            leaf->name.empty() ? "<param>" : leaf->name,
            data_type_to_string(leaf->dtype),
            tensor_dims_to_string(leaf)
        );

        ofs << "    \"" << node_id << "\" [label=\"" << label << "\", style=filled, fillcolor=lightgray];\n";
    }

    // 导出所有计算节点
    ofs << "\n    // Computation nodes\n";
    for (const auto* node : m_nodes) {
        if (visited.find(node) != visited.end()) continue;
        visited.insert(node);

        std::string node_id = std::format("tensor_0x{:x}", reinterpret_cast<uintptr_t>(node));
        std::string label = std::format(
            "{}\n{}\n{}",
            node->name.empty() ? operation_type_to_string(node->op_type) : node->name,
            data_type_to_string(node->dtype),
            tensor_dims_to_string(node)
        );

        ofs << "    \"" << node_id << "\" [label=\"" << label << "\"];\n";

        // 导出边（从源节点到当前节点）
        for (const auto* src : node->src) {
            if (src) {
                std::string src_id = std::format("tensor_0x{:x}", reinterpret_cast<uintptr_t>(src));
                ofs << "    \"" << src_id << "\" -> \"" << node_id << "\";\n";
            }
        }
    }

    ofs << "}\n";
    ofs.close();
}
