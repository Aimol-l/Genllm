#include <print>
#include <chrono>
#include <format>
#include "core/graph.h"
#include "backend/backend.h"
#include "core/gguf_parser.h"
#include "model/model.h"
#include "tokenizer.h"
#include "core/executor.h"
#include "utils/tools.hpp"

double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

int main(int argc, char** argv) {
    DeviceManager::instance().print_devices();

    const char* model_path = argc > 1 ? argv[1] : "models/Qwen3-4B-BF16.gguf";
    int gen_tokens = argc > 2 ? std::stoi(argv[2]) : 1024;

    GGUFParser parser(model_path);

    std::unique_ptr<ModelBase> model = ModelFactory::CreateFromGGUF(parser.info());

    ComputeGraph& graph = model->build_graph(parser.info());

    GraphScheduler::Config sched_cfg{
        .vocab_size = model->vocab_size(),
        .max_seq_len = 2048,
        .top_p = 1.0f,
        .temperature = 0.5f,
        .memory_headroom = 0.1f,
        .kv_cache_pool_factor = 1.05f,
        .activation_pool_factor = 1.05f
    };

    GraphScheduler scheduler(graph, sched_cfg);
    scheduler.schedule(DeviceManager::instance().get_devices());

    std::unique_ptr<MemoryManager>& manager = scheduler.mmanager();
    manager->load_weights(parser, scheduler.graph());

    Executor executor(scheduler);
    Tokenizer tokenizer = Tokenizer::from_gguf(parser);

    std::println("\nModel: {}", model_path);
    std::println("Output tokens: {}\n", gen_tokens);

    struct BenchPrompt {
        std::string name;
        std::string text;
    };

    std::vector<BenchPrompt> prompts = {
        {"数学",     "1+1="},
        {"常识",     "中国首都是哪里？"},
        {"代码",     "Write a Python function to check if a number is prime."},
        {"翻译",     "Translate into Chinese: Hello, how are you?"},
    };

    std::println("{:=^80}", " Benchmark ");
    std::println("{:<8} {:>6} {:>6} {:>12} {:>12} {:>10}","名称", "prompt", "生成", "耗时(ms)", "ms/token", "tokens/s");
    std::println("{:=^80}", "");

    for (auto& bp : prompts) {

        std::string chat_prompt = std::format("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", bp.text);

        std::vector<int32_t> prompt_ids = tokenizer.encode(chat_prompt);


        // Benchmark
        double t0 = now_sec();
        auto output = executor.generate(prompt_ids, gen_tokens, tokenizer.eos_id(), nullptr);
        
        double elapsed_ms = (now_sec() - t0) * 1000.0;

        double ms_per_tok = elapsed_ms / output.size();
        double tok_per_sec = 1000.0 / ms_per_tok;

        std::println("{:<8} {:>5} {:>6} {:>12.1f} {:>12.2f} {:>10.1f}",
            bp.name,
            static_cast<int>(prompt_ids.size()),
            output.size(),
            elapsed_ms,
            ms_per_tok,
            tok_per_sec
        );

        manager->reset_kv_cache();
    }

    std::println("{:=^80}\n", "");
    return 0;
}
