#include "core/graph.hpp"
#include "backend/backend.h"
#include "gguf_parser.h"
#include "model/model.h"
#include "tokenizer.h"
#include "core/executor.h"
#include <print>

int main() {

    DeviceManager::instance().print_devices();

    GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");

    // parser.info().print_info();

    std::unique_ptr<ModelBase> model = ModelFactory::CreateFromGGUF(parser.info());

    ComputeGraph& graph = model->build_graph(parser.info()); // 目前是batch固定1，seq_len动态的。


    GraphScheduler::Config sched_cfg{
        .vocab_size = model->vocab_size(),
        .kv_cache_per_layer = 0,    // 目前不区分层，统一估算一个值。实际实现时可以根据模型结构区分不同层的KV cache需求。
        .max_seq_len = 2048,        // 一个合理的默认值，实际使用时可以根据模型和需求调整
        .top_p = 0.8f,              // 采样时的 top-p 参数，越大越保守，越小越激进
        .temperature = 0.2f,        // 采样时的 temperature 越大越随机，越小越确定。0 表示贪心采样。
        .memory_headroom = 0.1f,        // 内存头部空间，预留给临时峰值，避免频繁 OOM
        .activation_pool_factor = 1.2f  // 激活内存池大小 = 实际激活内存需求 * activation_pool_factor。比实际需求大一点点，避免频繁 OOM
    };

    GraphScheduler scheduler(graph, sched_cfg);

    scheduler.schedule(DeviceManager::instance().get_devices());

    // scheduler.export_dot("qwen3_graph.dot");

    std::unique_ptr<MemoryManager>& manager = scheduler.mmanager();

    manager->load_weights(parser, scheduler.graph());

    Executor executor(scheduler);

    Tokenizer tokenizer = Tokenizer::from_gguf(parser);
    std::vector<int32_t> prompt_ids = tokenizer.encode("1+1=");
    std::println("Prompt IDs: {}", prompt_ids);

    try {

        std::vector<int32_t> output = executor.generate(prompt_ids, 10);

        std::string gen = tokenizer.decode(output);

        std::println("Output IDs: {}", output);
        std::println("Generated: \"{}\"", gen);

    } catch (const std::exception& e) {

        std::println("Executor error: {}", e.what());
    }

    return 0;
}
