#include <queue>
#include <format>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstring>
#include "core/tokenizer.h"

// ========== GPT-2 byte ↔ unicode 映射 ==========

// GPT-2 把每个 byte (0-255) 映射到一个可见 unicode 字符：
//   - 33-126 (!~), 161-172 (¡¬), 174-255 (®ÿ) 保持不变
//   - 其余 byte (0-32, 127-160, 173) 映射到 U+0100 开始的连续 codepoint
static uint32_t g_byte_to_unicode[256];
static uint8_t  g_unicode_to_byte[512]; // 只用到 index 256-511

static void init_byte_unicode_table() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    // 直接映射的字节范围
    std::vector<int> direct;
    for (int b = 33; b <= 126; ++b) direct.push_back(b);
    for (int b = 161; b <= 172; ++b) direct.push_back(b);
    for (int b = 174; b <= 255; ++b) direct.push_back(b);

    // 标记直接映射
    for (int b : direct) {
        g_byte_to_unicode[b] = static_cast<uint32_t>(b);
    }
    // 非直接映射的字节 → 从 U+0100 开始
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool is_direct = false;
        for (int d : direct) { if (d == b) { is_direct = true; break; } }
        if (!is_direct) {
            g_byte_to_unicode[b] = 256 + n;
            g_unicode_to_byte[256 + n] = static_cast<uint8_t>(b);
            ++n;
        }
    }
    // 直接映射的反向表
    for (int b : direct) {
        g_unicode_to_byte[b] = static_cast<uint8_t>(b);
    }
}

// 把原始字节序列转为 GPT-2 unicode 字符串
static std::string bytes_to_unicode(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() * 2); // worst case
    for (uint8_t b : raw) {
        uint32_t cp = g_byte_to_unicode[b];
        // encode codepoint to UTF-8
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return out;
}

// 把 GPT-2 unicode 字符串还原为原始字节
static std::string unicode_to_bytes(const std::string& mapped) {
    std::string out;
    out.reserve(mapped.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(mapped.data());
    const uint8_t* end = p + mapped.size();
    while (p < end) {
        uint32_t cp;
        if (*p < 0x80) {
            cp = *p; p += 1;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6; cp |= (p[1] & 0x3F); p += 2;
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12; cp |= (p[1] & 0x3F) << 6; cp |= (p[2] & 0x3F); p += 3;
        } else {
            cp = (*p & 0x07) << 18; cp |= (p[1] & 0x3F) << 12; cp |= (p[2] & 0x3F) << 6; cp |= (p[3] & 0x3F); p += 4;
        }
        out += static_cast<char>(g_unicode_to_byte[cp & 0x1FF]);
    }
    return out;
}

// ========== UTF-8 工具 ==========

static int utf8_char_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static bool is_alpha(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static bool is_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}
static bool is_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// 字母/数字/UTF-8多字节起始（视为"标识符字符"，类似 \p{L}|\p{N}）
static bool is_alnum_utf8(const char* s, const char* end) {
    if (s >= end) return false;
    uint8_t c = static_cast<uint8_t>(*s);
    if (is_alpha(c) || is_digit(c)) return true;
    return c >= 0xC0; // UTF-8 多字节起始字节 → 视为 letter
}

// ========== from_gguf ==========

Tokenizer Tokenizer::from_gguf(GGUFParser& parser) {
    init_byte_unicode_table();

    Tokenizer tok;
    auto& meta = parser.info().metadata;

    if (meta.contains("tokenizer.ggml.pre"))
        tok.pre_type_ = meta["tokenizer.ggml.pre"].get<std::string>();

    auto get_i64 = [&](const char* key) -> int32_t {
        if (meta.contains(key)) {
            auto& v = meta[key];
            if (v.is_number()) return static_cast<int32_t>(v.get<int64_t>());
            if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
        }
        return -1;
    };

    tok.bos_id_ = get_i64("tokenizer.ggml.bos_token_id");
    tok.eos_id_ = get_i64("tokenizer.ggml.eos_token_id");
    tok.pad_id_ = get_i64("tokenizer.ggml.padding_token_id");
    tok.add_bos_ = get_i64("tokenizer.ggml.add_bos") == 1;
    tok.add_eos_ = get_i64("tokenizer.ggml.add_eos") == 1;

    // load vocab (已在 GPT-2 unicode 表示下)
    if (!meta.contains("tokenizer.ggml.tokens") || !meta["tokenizer.ggml.tokens"].is_array())
        throw std::runtime_error("Tokenizer: tokenizer.ggml.tokens not found");

    auto& tokens_arr = meta["tokenizer.ggml.tokens"];
    tok.vocab_.resize(tokens_arr.size());
    for (size_t i = 0; i < tokens_arr.size(); ++i) {
        tok.vocab_[i] = tokens_arr[i].get<std::string>();
        tok.vocab_map_[tok.vocab_[i]] = static_cast<int32_t>(i);
    }

    // load merges → bpe_ranks (也在 GPT-2 unicode 表示下)
    if (meta.contains("tokenizer.ggml.merges") && meta["tokenizer.ggml.merges"].is_array()) {
        auto& merges_arr = meta["tokenizer.ggml.merges"];
        for (size_t i = 0; i < merges_arr.size(); ++i) {
            std::string merge = merges_arr[i].get<std::string>();
            auto sp = merge.find(' ');
            if (sp == std::string::npos) continue;
            tok.bpe_ranks_[{merge.substr(0, sp), merge.substr(sp + 1)}] = static_cast<int32_t>(i);
        }
    }

    return tok;
}

// ========== pre_tokenize ==========

// Qwen2 / GPT-2 风格的 pre-tokenization。
// 核心规则（regex 等价）：
//   [^\r\n\p{L}\p{N}]?\p{L}+   可选一个非字母数字非换行 + 字母序列
//   \p{N}+                       数字序列
//    ?[^\s\p{L}\p{N}]+[\r\n]*   可选空格 + 标点 + 可选拖尾换行
//   \s*[\r\n]+                   空白 + 换行
//   \s+(?!\S)                    拖尾空白
//   \s+                          其余空白
// 关键：空格被附在前面的 word/punctuation 片段上（由第一条和第三条规则捕获）

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    const char* p = text.data();
    const char* end = p + text.size();

    auto emit = [&](const char* from, const char* to) {
        if (from < to) pieces.emplace_back(from, to - from);
    };

    while (p < end) {
        const char* start = p;

        // A) [^\r\n\p{L}\p{N}]?\p{L}+  — 可选前导字符 + 字母序列
        //    前导字符可以是空格、标点等（但不是换行/字母/数字）
        {
            const char* q = p;
            bool has_leader = false;
            if (q < end && !is_space(*q) && *q != '\r' && *q != '\n' && !is_alnum_utf8(q, end)) {
                q += utf8_char_len(*q);
                has_leader = true;
            }
            // 如果 q 没前进（没有 leader），检查 q 位置是否是字母
            // 如果 q 前进了但有 leader，检查后续是否是字母
            const char* letter_start = q;
            while (q < end && is_alnum_utf8(q, end) && !is_digit(*q)) {
                // 排除纯数字（数字由单独规则处理）
                q += utf8_char_len(*q);
            }
            // 检查我们是否真的匹配了字母（至少一个）
            if (q > letter_start && is_alpha(*letter_start)) {
                emit(start, q);
                p = q;
                continue;
            }
            if (q > letter_start && static_cast<uint8_t>(*letter_start) >= 0xC0) {
                // UTF-8 多字节作为"字母"
                emit(start, q);
                p = q;
                continue;
            }
        }

        // B) 可选空格 + 字母序列 (空格作为 leader)
        if (*p == ' ' && p + 1 < end) {
            const char* after_space = p + 1;
            uint8_t ac = static_cast<uint8_t>(*after_space);
            if (is_alpha(ac) || ac >= 0xC0) {
                const char* q = after_space;
                while (q < end && (is_alpha(static_cast<uint8_t>(*q)) || static_cast<uint8_t>(*q) >= 0xC0))
                    q += utf8_char_len(static_cast<uint8_t>(*q));
                if (q > after_space) {
                    emit(start, q);
                    p = q;
                    continue;
                }
            }
        }

        // C) \p{N}+ — 数字序列
        if (is_digit(*p)) {
            while (p < end && is_digit(*p)) ++p;
            emit(start, p);
            continue;
        }

        // D) 可选空格 + 标点 + 可选拖尾换行
        if (*p == ' ' || !is_space(*p)) {
            const char* q = p;
            if (*q == ' ') ++q; // 消耗可选前导空格
            const char* punct_start = q;
            while (q < end && !is_space(*q) && !is_alnum_utf8(q, end)) {
                q += utf8_char_len(*q);
            }
            if (q > punct_start) {
                // 消耗拖尾换行
                while (q < end && (*q == '\r' || *q == '\n')) ++q;
                emit(start, q);
                p = q;
                continue;
            }
        }

        // E) 换行
        if (*p == '\n' || *p == '\r') {
            while (p < end && (*p == '\n' || *p == '\r')) ++p;
            emit(start, p);
            continue;
        }

        // F) 其余空白
        if (is_space(*p)) {
            while (p < end && is_space(*p) && *p != '\n' && *p != '\r') ++p;
            emit(start, p);
            continue;
        }

        // G) 单字符 fallback
        p += utf8_char_len(*p);
        emit(start, p);
    }

    return pieces;
}

// ========== BPE ==========

struct Symbol {
    std::string text;
    int prev = -1;
    int next = -1;
};

std::vector<int32_t> Tokenizer::bpe(const std::string& word) const {
    // word 已经过 bytes_to_unicode 映射
    if (word.empty()) return {};

    // 拆分为单个 UTF-8 字符
    std::vector<Symbol> symbols;
    const char* p = word.data();
    const char* wend = p + word.size();
    while (p < wend) {
        int len = utf8_char_len(*p);
        symbols.push_back({std::string(p, len),
                           static_cast<int>(symbols.size()) - 1,
                           static_cast<int>(symbols.size()) + 1});
        p += len;
    }
    if (!symbols.empty()) {
        symbols.front().prev = -1;
        symbols.back().next = -1;
    }

    if (symbols.size() == 1) {
        auto it = vocab_map_.find(symbols[0].text);
        return (it != vocab_map_.end()) ? std::vector<int32_t>{it->second} : std::vector<int32_t>{};
    }

    // 优先队列：min-heap by rank
    using Bigram = std::pair<int32_t, std::pair<int, int>>;
    std::priority_queue<Bigram, std::vector<Bigram>, std::greater<Bigram>> queue;

    // 初始 bigrams
    for (int i = 0; i < static_cast<int>(symbols.size()) - 1; ++i) {
        auto it = bpe_ranks_.find({symbols[i].text, symbols[i + 1].text});
        if (it != bpe_ranks_.end()) queue.push({it->second, {i, i + 1}});
    }

    while (!queue.empty()) {
        auto [rank, pair] = queue.top();
        queue.pop();
        int left = pair.first, right = pair.second;

        if (left < 0 || left >= static_cast<int>(symbols.size())) continue;
        if (right < 0 || right >= static_cast<int>(symbols.size())) continue;
        if (symbols[left].next != right) continue;

        auto it = bpe_ranks_.find({symbols[left].text, symbols[right].text});
        if (it == bpe_ranks_.end() || it->second != rank) continue;

        // 合并 right → left
        symbols[left].text += symbols[right].text;
        symbols[left].next = symbols[right].next;
        if (symbols[right].next >= 0)
            symbols[symbols[right].next].prev = left;

        // 新 bigrams
        if (symbols[left].prev >= 0) {
            int pv = symbols[left].prev;
            auto pit = bpe_ranks_.find({symbols[pv].text, symbols[left].text});
            if (pit != bpe_ranks_.end()) queue.push({pit->second, {pv, left}});
        }
        if (symbols[left].next >= 0) {
            int nx = symbols[left].next;
            auto nit = bpe_ranks_.find({symbols[left].text, symbols[nx].text});
            if (nit != bpe_ranks_.end()) queue.push({nit->second, {left, nx}});
        }
    }

    // 沿链表遍历存活的 symbol，转换为 token IDs
    std::vector<int32_t> result;
    int head = 0; // 第一个 symbol 的 prev == -1
    for (int i = head; i >= 0 && i < static_cast<int>(symbols.size()); ) {
        auto it = vocab_map_.find(symbols[i].text);
        if (it != vocab_map_.end()) result.push_back(it->second);
        i = symbols[i].next;
    }
    return result;
}

// ========== encode ==========

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> tokens;

    if (add_bos_ && bos_id_ >= 0)
        tokens.push_back(bos_id_);

    auto pieces = pre_tokenize(text);
    for (auto& piece : pieces) {
        // 把原始字节转为 GPT-2 unicode 表示后再 BPE
        std::string mapped = bytes_to_unicode(piece);
        auto ids = bpe(mapped);
        tokens.insert(tokens.end(), ids.begin(), ids.end());
    }

    if (add_eos_ && eos_id_ >= 0)
        tokens.push_back(eos_id_);

    return tokens;
}

// ========== decode ==========

std::string Tokenizer::token_to_text(int32_t id) const {
    if (id < 0 || id >= static_cast<int32_t>(vocab_.size())) return "";

    const std::string& tok = vocab_[id];

    // 特殊 byte token: <0xHH>
    if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
        uint8_t byte = 0;
        for (int i = 3; i < 5; ++i) {
            byte <<= 4;
            if (tok[i] >= '0' && tok[i] <= '9') byte |= (tok[i] - '0');
            else if (tok[i] >= 'a' && tok[i] <= 'f') byte |= (tok[i] - 'a' + 10);
            else if (tok[i] >= 'A' && tok[i] <= 'F') byte |= (tok[i] - 'A' + 10);
        }
        return std::string(1, static_cast<char>(byte));
    }

    // 常规 token：从 GPT-2 unicode 还原为原始字节
    return unicode_to_bytes(tok);
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string result;
    for (int32_t id : ids) {
        if (id == eos_id_) break;
        result += token_to_text(id);
    }
    return result;
}
