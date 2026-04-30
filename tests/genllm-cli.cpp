#include <iostream>
#include <print>
#include <format>
#include <string>
#include <vector>
#include "core/graph.h"
#include "backend/backend.h"
#include "core/gguf_parser.h"
#include "model/model.h"
#include "tokenizer.h"
#include "core/executor.h"

struct CliOptions {
    std::string model_path;
    std::string prompt;
    std::string system_prompt = "你是一个日常对话AI助手,回复简洁准确即可。";
    int max_tokens = 512;
    float temperature = 0.5f;
    float top_p = 0.9f;
    int max_seq_len = 2048;
    bool interactive = false;
    bool show_help = false;
};

static CliOptions parse_args(int argc, char** argv) {
    CliOptions opts;
    auto require_arg = [&](int& i, const char* flag) {
        if (++i >= argc) {
            std::println(stderr, "Error: {} requires an argument", flag);
            std::exit(1);
        }
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            opts.show_help = true;
        } else if (arg == "-m" || arg == "--model") {
            opts.model_path = require_arg(i, arg.c_str());
        } else if (arg == "-p" || arg == "--prompt") {
            opts.prompt = require_arg(i, arg.c_str());
        } else if (arg == "-n" || arg == "--max-tokens") {
            try { opts.max_tokens = std::stoi(require_arg(i, arg.c_str())); }
            catch (...) { std::println(stderr, "Error: invalid value for {}", arg); std::exit(1); }
        } else if (arg == "--temp") {
            try { opts.temperature = std::stof(require_arg(i, arg.c_str())); }
            catch (...) { std::println(stderr, "Error: invalid value for {}", arg); std::exit(1); }
        } else if (arg == "--top-p") {
            try { opts.top_p = std::stof(require_arg(i, arg.c_str())); }
            catch (...) { std::println(stderr, "Error: invalid value for {}", arg); std::exit(1); }
        } else if (arg == "--max-seq") {
            try { opts.max_seq_len = std::stoi(require_arg(i, arg.c_str())); }
            catch (...) { std::println(stderr, "Error: invalid value for {}", arg); std::exit(1); }
        } else if (arg == "-s" || arg == "--system") {
            opts.system_prompt = require_arg(i, arg.c_str());
        } else if (arg == "-i" || arg == "--interactive") {
            opts.interactive = true;
        } else if (opts.model_path.empty()) {
            opts.model_path = arg;
        } else if (opts.prompt.empty()) {
            opts.prompt = arg;
        }
    }
    return opts;
}

static void print_help(const char* prog) {
    std::println("Usage: {} [options] <model.gguf> [prompt]", prog);
    std::println("");
    std::println("Options:");
    std::println("  -m, --model <path>      GGUF model file path");
    std::println("  -p, --prompt <text>     Input prompt");
    std::println("  -s, --system <text>     System prompt (default: 你是一个有用的AI助手。)");
    std::println("  -n, --max-tokens <N>    Maximum tokens to generate (default: 512)");
    std::println("  --temp <T>              Sampling temperature (default: 0.5, 0=greedy)");
    std::println("  --top-p <P>             Top-p sampling threshold (default: 0.9)");
    std::println("  --max-seq <N>           Max sequence length (default: 2048)");
    std::println("  -i, --interactive       Interactive chat mode (multi-turn)");
    std::println("  -h, --help              Show this help");
}

int main(int argc, char** argv) {
    auto opts = parse_args(argc, argv);

    if (opts.show_help || opts.model_path.empty()) {
        print_help(argv[0]);
        return opts.show_help ? 0 : 1;
    }

    if (opts.prompt.empty() && !opts.interactive) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!opts.prompt.empty()) opts.prompt += "\n";
            opts.prompt += line;
        }
        if (opts.prompt.empty()) {
            std::println(stderr, "Error: no prompt provided");
            return 1;
        }
    }

    // DeviceManager::instance().print_devices();

    // ---- 加载模型 ----
    GGUFParser parser(opts.model_path);
    std::unique_ptr<ModelBase> model = ModelFactory::CreateFromGGUF(parser.info());
    ComputeGraph& graph = model->build_graph(parser.info());

    GraphScheduler::Config sched_cfg{
        .vocab_size = model->vocab_size(),
        .max_seq_len = static_cast<int64_t>(opts.max_seq_len),
        .top_p = opts.top_p,
        .temperature = opts.temperature,
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

    // ---- 单轮模式 ----
    if (!opts.interactive) {
        std::string chat_prompt = std::format("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", opts.prompt);
        std::vector<int32_t> prompt_ids = tokenizer.encode(chat_prompt);

        std::println(stderr, "\nGenerating {} tokens max...\n", opts.max_tokens);
        try {
            executor.generate(prompt_ids, opts.max_tokens, tokenizer.eos_id(), &tokenizer);
        } catch (const std::exception& e) {
            std::println(stderr, "Error: {}", e.what());
            return 1;
        }
        return 0;
    }

    // ---- 交互模式（多轮，保留 KV cache）----
    std::println(stderr, "\nInteractive mode. Type your messages. Ctrl+D or empty line to exit.\n");

    // 系统 prompt
    std::vector<int32_t> system_ids = tokenizer.encode(std::format("<|im_start|>system\n{}<|im_end|>\n", opts.system_prompt));
    executor.append_tokens(system_ids);

    while (true) {
        std::print(">>> ");
        std::fflush(stdout);

        std::string user_msg;
        if (!std::getline(std::cin, user_msg) || user_msg.empty()) break;

        // 检查上下文长度
        if (executor.seq_pos() >= opts.max_seq_len - opts.max_tokens) {
            std::println(stderr, "\n[Warning] Context length {} approaching max_seq_len {}. Exiting to avoid overflow.\n",executor.seq_pos(), opts.max_seq_len);
            break;
        }

        // user 消息
        std::vector<int32_t> user_ids = tokenizer.encode(std::format("<|im_start|>user\n{}<|im_end|>\n", user_msg));
        executor.append_tokens(user_ids);

        // assistant 头
        std::vector<int32_t> asst_header = tokenizer.encode("<|im_start|>assistant\n");
        executor.append_tokens(asst_header);

        // 生成回复
        std::vector<int32_t> output;
        for (int i = 0; i < opts.max_tokens; ++i) {
            int32_t next = executor.sample();
            if (next == tokenizer.eos_id()) break;

            std::string token_str = tokenizer.decode({next});
            std::print("{}", token_str);
            std::fflush(stdout);

            output.push_back(next);
            executor.decode_step(next);
        }
        std::print("\n\n");
    }

    std::println(stderr, "\nBye!");
    return 0;
}
