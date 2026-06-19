#include <print>

#include "tokenizer.h"
#include "model/model.h"
#include "utils/tools.hpp"
#include "core/executor.h"
#include "backend/backend.h"
#include "core/gguf_parser.h"

int main() {

    DeviceManager::instance().print_devices();

    GGUFParser parser("weights/Qwen3-0.6B-BF16.gguf");

    parser.info().print_info();

    std::unique_ptr<ModelBase> model = ModelFactory::CreateFromGGUF(parser.info());

    auto graph = model->build_graph(parser.info()); // 目前是batch固定1，seq_len动态的。

    // graph->export_dot("docs/qwen3-graph-abs.dot");

    GraphScheduler::Config sched_cfg{
        .vocab_size = model->vocab_size(),
        .max_seq_len = 8192,            // 上下文长度。主要影响激活池大小，kv-cache池影响有限
        .top_p = 0.9f,                  // 采样时的 top-p 参数，越大越保守，越小越激进
        .temperature = 0.5f,            // 采样时的 temperature 越大越随机，越小越确定。0 表示贪心采样。
        .memory_headroom = 0.05f,        // 内存头部空间，预留给临时峰值，避免频繁 OOM
        .kv_cache_pool_factor = 1.0f,  // 缓存内存池大小 = 实际激活内存需求 * activation_pool_factor。比实际需求大一点点，避免可能的OOM
        .activation_pool_factor = 1.0f // 激活内存池大小 = 实际激活内存需求 * activation_pool_factor。比实际需求大一点点，避免可能的OOM
    };

    GraphScheduler scheduler(std::move(graph), sched_cfg);

    scheduler.schedule(DeviceManager::instance().get_devices());

    // scheduler.export_dot("docs/qwen3-graph-sch.dot");

    std::unique_ptr<MemoryManager>& manager = scheduler.mmanager();

    manager->load_weights(parser, scheduler.graph()); // 加载权重

    Executor executor(scheduler);

    Tokenizer tokenizer = Tokenizer::from_gguf(parser);

    const std::string user_prompt = "中国首都是哪里？";
    std::string chat_prompt = model->chat_template(user_prompt,true);

    std::vector<int32_t> tokens = tokenizer.encode(chat_prompt);

    std::println("tokens={}",tokens);

    try {

        RUNNING_TIME(executor.generate(tokens, 512, tokenizer.eos_id(), &tokenizer));

    } catch (const std::exception& e) {

        std::println("Executor error: {}", e.what());
    }
    return 0;
}