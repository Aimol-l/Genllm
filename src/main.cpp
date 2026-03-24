#include <iostream>
#include "graph.h"
#include "scheduler.h"
#include "executor.h"
#include "backend.h"
#include "context.hpp"
#include "memory.hpp"
#include "gguf_parser.h"
#include "model/model.h"


int main() {
    try {
        // ==================== 0. 检测可用设备 ====================
        std::println("=== Step 0: Detecting available devices ===");
        print_available_devices();

        // ==================== 1. 解析GGUF文件 ====================
        std::println("\n=== Step 1: Parsing GGUF file ===");
        GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");
        GGUFInfo info = parser.parse();
        info.print_info();

        // ==================== 2. 创建模型 ====================
        std::println("\n=== Step 2: Creating model ===");
        auto model = ModelFactory::CreateFromGGUF(info);
        std::println("Created model: {}", model->name);

        // ==================== 3. 构建计算图 ====================
        std::println("\n=== Step 3: Building computation graph ===");
        ComputeGraph& graph = model->build_graph(info);
        
        // ==================== 4. 创建调度器 ====================
        std::println("\n=== Step 4: Creating scheduler ===");

        // 方式1：自动使用所有可用设备
        // Scheduler scheduler(&model->graph);

        // 方式2：创建CPU专用调度器
        // auto scheduler = create_cpu_scheduler(&model->graph);

        // 方式3：创建GPU优先调度器（如果有GPU）
        auto scheduler = create_gpu_scheduler(graph);

        // 调度器会自动使用 DeviceManager 检测到的所有设备
        // 不需要手动添加设备！

        // ==================== 5. 图切分 ====================
        std::println("\n=== Step 5: Partitioning graph ===");
        scheduler.partition_graph();
        scheduler.print_subgraphs();

        // ==================== 6. 内存估算 ====================
        std::println("\n=== Step 6: Estimating memory ===");
        scheduler.print_memory_stats();

        // ==================== 7. 构建执行计划 ====================
        std::println("\n=== Step 7: Building execute order ===");
        scheduler.build_execute_order(graph_root);
        scheduler.optimize_execute_order();
        scheduler.print_execute_order();

        // 导出调度计划（可选）
        // scheduler.export_schedule_dot("schedule.dot");

        // ==================== 8. 创建内存池并分配权重内存 ====================
        std::println("\n=== Step 8: Creating memory pools and allocating weights ===");

        // 获取调度器的内存管理器
        MemoryManager* mem_manager = scheduler.get_memory_manager();

        // 为所有子图分配权重内存
        scheduler.allocate_all_weights();

        // ==================== 9. 加载权重数据 ====================
        std::println("\n=== Step 9: Loading weight data from GGUF ===");

        // 使用内存管理器加载权重（从 GGUF 文件读取数据到已分配的内存）
        model->load_weights(info, mem_manager);

        // ==================== 10. 自回归推理 ====================
        std::println("\n=== Step 10: Autoregressive inference ===");

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
