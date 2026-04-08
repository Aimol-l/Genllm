#include "graph.hpp"
#include "backend/backend.h"
#include "gguf_parser.h"
#include "model/model.h"
#include "core/scheduler.h"

int main() {
    DeviceManager::instance().print_devices();

    GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");
    GGUFInfo info = parser.parse();
    // info.print_info();

    auto model = ModelFactory::CreateFromGGUF(info);
    
    ComputeGraph& graph = model->build_graph(info);

    GraphScheduler scheduler(graph);
    
    scheduler.schedule(DeviceManager::instance().get_devices());

    scheduler.export_dot("qwen3_graph.dot");


//     scheduler.allocate_all_weights();   // 预分配权重内存，减少后续迁移开销

//     // 获取调度器的内存管理器
//     Executor executor(&scheduler);
    
//     std::vector<int> prompt = {12345, 67890}; // 示例token IDs
//     int max_tokens = 100;
//     // TODO: 创建输入张量并开始推理循环
//     for (int step = 0; step < 1; ++step) {  // 暂时只执行1步用于测试
//         std::println("\n--- Step {} ---", step);

//         // 执行前向传播（按照拓扑序执行，不是按子图顺序！）
//         executor.execute(graph_root);

//         // 采样下一个token
//         // int next_token = sample_from_logits(graph_root);

//         // 更新输入
//         // input_ids = next_token;

//         // 清理临时张量
//         // scheduler.cleanup_temp_tensors();

//         // 检查结束条件
//         // if (next_token == eos_token) break;
//     }

//     std::println("\n=== Inference completed ===");
    return 0;
}
