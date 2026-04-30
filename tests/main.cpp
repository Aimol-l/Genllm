#include <print>
#include "core/graph.h"
#include "backend/backend.h"
#include "core/gguf_parser.h"
#include "model/model.h"
#include "tokenizer.h"
#include "core/executor.h"
#include "utils/tools.hpp"
int main() {

    DeviceManager::instance().print_devices();

    GGUFParser parser("models/Qwen3-4B-BF16.gguf");

    parser.info().print_info();

    std::unique_ptr<ModelBase> model = ModelFactory::CreateFromGGUF(parser.info());

    ComputeGraph& graph = model->build_graph(parser.info()); // 目前是batch固定1，seq_len动态的。

    GraphScheduler::Config sched_cfg{
        .vocab_size = model->vocab_size(),
        .max_seq_len = 8192,            // 上下文长度。主要影响激活池大小，kv-cache池影响有限
        .top_p = 0.9f,                  // 采样时的 top-p 参数，越大越保守，越小越激进
        .temperature = 0.5f,            // 采样时的 temperature 越大越随机，越小越确定。0 表示贪心采样。
        .memory_headroom = 0.05f,        // 内存头部空间，预留给临时峰值，避免频繁 OOM
        .kv_cache_pool_factor = 1.05f,  // 缓存内存池大小 = 实际激活内存需求 * activation_pool_factor。比实际需求大一点点，避免可能的OOM
        .activation_pool_factor = 1.05f // 激活内存池大小 = 实际激活内存需求 * activation_pool_factor。比实际需求大一点点，避免可能的OOM
    };

    GraphScheduler scheduler(graph, sched_cfg);

    scheduler.schedule(DeviceManager::instance().get_devices());

    // scheduler.export_dot("qwen3-graph.dot");

    std::unique_ptr<MemoryManager>& manager = scheduler.mmanager();

    manager->load_weights(parser, scheduler.graph());

    Executor executor(scheduler);

    Tokenizer tokenizer = Tokenizer::from_gguf(parser);

    const std::string user_msg = "1+1=";

    std::string chat_prompt = std::format("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_msg);

    std::vector<int32_t> prompt_ids = tokenizer.encode(chat_prompt);

    try {

        RUNNING_TIME(executor.generate(prompt_ids, 512, tokenizer.eos_id(), &tokenizer));

    } catch (const std::exception& e) {

        std::println("Executor error: {}", e.what());
    }
    return 0;
}