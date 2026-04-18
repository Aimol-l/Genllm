// 临时工具：dump GGUF tokenizer 相关的 metadata key
#include "gguf_parser.h"
#include <print>

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "models/Qwen3-0.6B-BF16.gguf";
    GGUFParser parser(path);
    auto& meta = parser.info().metadata;

    std::vector<std::string> keys = {
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.add_bos",
        "tokenizer.ggml.add_eos",
        "tokenizer.ggml.add_prefix",
        "tokenizer.ggml.remove_extra_ws",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unk_id",
        "tokenizer.ggml.sep_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.score_threshold",
    };

    for (auto& key : keys) {
        if (!meta.contains(key)) {
            std::println("  {} = (not found)", key);
            continue;
        }
        auto& val = meta[key];
        if (val.is_string()) {
            std::string s = val.get<std::string>();
            if (s.size() > 200) s = s.substr(0, 200) + "...";
            std::println("  {} = \"{}\"", key, s);
        } else if (val.is_boolean()) {
            std::println("  {} = {}", key, val.get<bool>());
        } else if (val.is_number()) {
            std::println("  {} = {}", key, val.get<int64_t>());
        } else if (val.is_array()) {
            std::println("  {} = [array, {} elements]", key, val.size());
        } else {
            std::println("  {} = (unknown type)", key);
        }
    }

    // vocab size
    for (auto& key : {"tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.scores",
                       "tokenizer.ggml.token_type", "tokenizer.ggml.leading_space",
                       "tokenizer.ggml.trailing_space", "tokenizer.ggml.json_template"}) {
        if (meta.contains(key) && meta[key].is_array()) {
            std::println("  {} = [array, {} elements]", key, meta[key].size());
        } else if (meta.contains(key) && meta[key].is_string()) {
            std::println("  {} = \"{}\"", key, meta[key].get<std::string>());
        }
    }
    return 0;
}
