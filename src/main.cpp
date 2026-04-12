#include "graph.hpp"
#include "backend/backend.h"
#include "gguf_parser.h"
#include "model/model.h"
#include "core/scheduler.h"
#include "core/executor.h"

int main() {
    DeviceManager::instance().print_devices();

    GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");

    parser.info().print_info();

    std::unique_ptr<ModelBase> model = ModelFactory::CreateFromGGUF(parser.info());

    ComputeGraph& graph = model->build_graph(parser.info());

    constexpr int64_t runtime_max_seq = 2048; // 一个合理的默认值，实际使用时可以根据模型和需求调整

    GraphScheduler::Config sched_cfg(0.1, 0, runtime_max_seq, 1.2f);

    GraphScheduler scheduler(graph, sched_cfg);

    scheduler.schedule(DeviceManager::instance().get_devices());

    scheduler.export_dot("qwen3_graph.dot");

    std::unique_ptr<MemoryManager>& manager = scheduler.mmanager();

    manager->load_weights(parser, scheduler.graph());

    // ========== 自回归生成 ==========
    Executor executor(scheduler);

    std::vector<int32_t> prompt = {151644, 872, 198};

    try {
        auto output = executor.generate(prompt, 10);

        std::println("Generated {} tokens", output.size());
    } catch (const std::exception& e) {
        std::println("Executor error: {}", e.what());
    }
    // 把output解析成文本（需要tokenizer，这里暂时不实现）

    // std::string output_text = "[decoded text here]";


    scheduler.mmanager()->print_all_usage();
    return 0;
}
