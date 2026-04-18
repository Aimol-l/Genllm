#pragma once

#include "gguf_parser.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>

class Tokenizer {
private:
    std::vector<std::string> vocab_;   // id -> token text
    std::unordered_map<std::string, int32_t> vocab_map_; // token text -> id
    std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks_; // merge priority

    int32_t eos_id_ = -1;
    int32_t bos_id_ = -1;
    int32_t pad_id_ = -1;
    bool add_bos_ = false;
    bool add_eos_ = false;
    std::string pre_type_; // "qwen2", "llama3", etc.

public:
    // 从 GGUF 文件加载词表
    static Tokenizer from_gguf(GGUFParser& parser);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t eos_id() const { return eos_id_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t vocab_size() const { return static_cast<int32_t>(vocab_.size()); }

private:
    // BPE
    std::vector<int32_t> bpe(const std::string& word) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;
    // Decode helpers
    std::string token_to_text(int32_t id) const;
};
