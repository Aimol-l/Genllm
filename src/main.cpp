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
        backend::init_builtin_backends();
        print_available_devices();
        GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");
        GGUFInfo info = parser.parse();
        info.print_info();

        auto model = ModelFactory::CreateFromGGUF(info);
        std::println("Created model: {}", model->name);

        ComputeGraph& graph = model->build_graph(info);
        GraphScheduler::Config config;
        config.weight_comm = 5.0;      // 高通信权重避免 Ping-pong
        config.weight_balance = 0.5;   // 低负载均衡避免过度分散
        GraphScheduler scheduler(config);
        scheduler.schedule(graph, get_available_devices());

        // 获取调度器的内存管理器
        MemoryManager* mem_manager = scheduler.get_memory_manager();
        scheduler.allocate_all_weights();
        model->load_weights(info, mem_manager);

        Executor executor(&scheduler);
        
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
