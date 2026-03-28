#include <iostream>
#include "graph.hpp"
#include "scheduler.h"
#include "executor.h"
#include "backend/backend.h"
#include "context.hpp"
#include "memory.hpp"
#include "gguf_parser.h"
#include "model/model.h"


int main() {
    try {
        // ==================== 0. 初始化后端 ====================
        std::println("=== Step 0: Initializing backends ===");
        backend::init_builtin_backends();

        // ==================== 1. 检测可用设备 ====================
        std::println("\n=== Step 1: Detecting available devices ===");
        print_available_devices();

        // ==================== 2. 解析GGUF文件 ====================
        std::println("\n=== Step 2: Parsing GGUF file ===");
        GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");
        GGUFInfo info = parser.parse();
        info.print_info();

        // ==================== 3. 创建模型 ====================
        std::println("\n=== Step 3: Creating model ===");
        auto model = ModelFactory::CreateFromGGUF(info);
        std::println("Created model: {}", model->name);

        // ==================== 4. 构建计算图 ====================
        std::println("\n=== Step 4: Building computation graph ===");
        ComputeGraph& graph = model->build_graph(info);

        // ==================== 5. 创建调度器 ====================
        std::println("\n=== Step 5: Creating scheduler ===");

        GraphScheduler scheduler; // 需要

        // ==================== 6. 图切分 ====================
        std::println("\n=== Step 6: Partitioning graph ===");
        scheduler.graph_backend_assignment(graph,get_available_devices());

        // 导出调度计划（可选）
        // scheduler.export_schedule_dot("schedule.dot");

        // ==================== 9. 创建内存池并分配权重内存 ====================
        std::println("\n=== Step 9: Creating memory pools and allocating weights ===");

        // 获取调度器的内存管理器
        MemoryManager* mem_manager = scheduler.get_memory_manager();

        // 为所有子图分配权重内存
        scheduler.allocate_all_weights();

        // ==================== 10. 加载权重数据 ====================
        std::println("\n=== Step 10: Loading weight data from GGUF ===");

        // 使用内存管理器加载权重（从 GGUF 文件读取数据到已分配的内存）
        model->load_weights(info, mem_manager);

        // ==================== 11. 自回归推理 ====================
        std::println("\n=== Step 11: Autoregressive inference ===");

        // 创建执行器
        Executor executor(&scheduler);

        // 准备输入
        std::vector<int> prompt = {12345, 67890}; // 示例token IDs
        int max_tokens = 100;

        // TODO: 创建输入张量并开始推理循环
        for (int step = 0; step < 1; ++step) {  // 暂时只执行1步用于测试
            std::println("\n--- Step {} ---", step);

            // 执行前向传播（按照拓扑序执行，不是按子图顺序！）
            executor.execute(graph_root);

            // 采样下一个token
            // int next_token = sample_from_logits(graph_root);

            // 更新输入
            // input_ids = next_token;

            // 清理临时张量
            // scheduler.cleanup_temp_tensors();

            // 检查结束条件
            // if (next_token == eos_token) break;
        }

        std::println("\n=== Inference completed ===");

    } catch (const std::exception& e) {
        std::println("Error: {}", e.what());
        return 1;
    }

    return 0;

}
