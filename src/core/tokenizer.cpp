#include <queue>
#include <format>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <uchar.h>
#include "core/tokenizer.h"

// ========== GPT-2 byte <-> unicode mapping ==========

static uint32_t g_byte_to_unicode[256];
static uint8_t  g_unicode_to_byte[65536];

static void init_byte_unicode_table() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    // 直接映射的字节范围 (与 GPT-2 一致)
    std::vector<int> direct;
    for (int b = 33; b <= 126; ++b) direct.push_back(b);
    for (int b = 161; b <= 172; ++b) direct.push_back(b);
    for (int b = 174; b <= 255; ++b) direct.push_back(b);

    std::fill(g_unicode_to_byte, g_unicode_to_byte + 65536, uint8_t(0xFF));

    for (int b : direct) {
        g_byte_to_unicode[b] = static_cast<uint32_t>(b);
        g_unicode_to_byte[b] = static_cast<uint8_t>(b);
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool is_direct = false;
        for (int d : direct) { if (d == b) { is_direct = true; break; } }
        if (!is_direct) {
            uint32_t cp = 256 + n;
            g_byte_to_unicode[b] = cp;
            g_unicode_to_byte[cp] = static_cast<uint8_t>(b);
            ++n;
        }
    }
}

static std::string bytes_to_unicode(const std::string& raw) {
    std::string out;
    out.reserve(raw.size() * 2);
    for (uint8_t b : raw) {
        uint32_t cp = g_byte_to_unicode[b];
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
        if (cp < 65536 && g_unicode_to_byte[cp] != 0xFF) {
            out += static_cast<char>(g_unicode_to_byte[cp]);
        }
    }
    return out;
}

// ========== UTF-8 helpers ==========

static int utf8_char_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static char32_t utf8_decode(const char* s, int& len) {
    uint8_t c = static_cast<uint8_t>(*s);
    if (c < 0x80) { len = 1; return c; }
    if ((c & 0xE0) == 0xC0) { len = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) { len = 3; return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); }
    len = 4; return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
}

static bool unicode_is_letter(char32_t cp) {
    return std::isalpha(cp);
}

static bool unicode_is_digit(char32_t cp) {
    return std::isdigit(cp);
}

static bool unicode_is_space(char32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}

static bool unicode_is_newline(char32_t cp) {
    return cp == '\n' || cp == '\r';
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

    if (!meta.contains("tokenizer.ggml.tokens") || !meta["tokenizer.ggml.tokens"].is_array())
        throw std::runtime_error("Tokenizer: tokenizer.ggml.tokens not found");

    auto& tokens_arr = meta["tokenizer.ggml.tokens"];
    tok.vocab_.resize(tokens_arr.size());
    for (size_t i = 0; i < tokens_arr.size(); ++i) {
        tok.vocab_[i] = tokens_arr[i].get<std::string>();
        tok.vocab_map_[tok.vocab_[i]] = static_cast<int32_t>(i);
    }

    if (meta.contains("tokenizer.ggml.merges") && meta["tokenizer.ggml.merges"].is_array()) {
        auto& merges_arr = meta["tokenizer.ggml.merges"];
        for (size_t i = 0; i < merges_arr.size(); ++i) {
            std::string merge = merges_arr[i].get<std::string>();
            auto sp = merge.find(' ');
            if (sp == std::string::npos) continue;
            tok.bpe_ranks_[{merge.substr(0, sp), merge.substr(sp + 1)}] = static_cast<int32_t>(i);
        }
    }

    for (size_t i = 0; i < tok.vocab_.size(); ++i) {
        if (tok.vocab_[i].empty()) continue;
        auto pieces = tok.pre_tokenize(tok.vocab_[i]);
        if (pieces.size() > 1) {
            tok.added_tokens_map_[tok.vocab_[i]] = static_cast<int32_t>(i);
        }
    }

    tok.added_tokens_.assign(tok.added_tokens_map_.begin(), tok.added_tokens_map_.end());
    std::sort(tok.added_tokens_.begin(), tok.added_tokens_.end(),
        [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    return tok;
}

// ========== pre_tokenize (state machine) ==========
//
// 实现 Qwen3/GPT-2 的完整正则:
//   '(?i:[sdmt]|ll|ve|re)             英文缩写
//   [^\r\n\p{L}\p{N}]?\p{L}+          可选前导非字母数字 + 字母
//   \p{N}{1,3}                        1-3位数字
//    ?[^\s\p{L}\p{N}]+[\r\n]*         可选空格 + 标点 + 拖尾换行
//   \s*[\r\n]+                        空白 + 换行
//   \s+(?!\S)                         拖尾空白
//   \s+                               其余空白

// 检查是否匹配英文缩写: 's, 't, 'd, 'm, 'S, 'T, 'D, 'M, 'll, 've, 're, 'LL, 'VE, 'RE
static int match_contraction(const char* p, const char* end) {
    // p 指向 "'" 字符
    if (p + 1 >= end) return 0;
    char next = p[1];
    switch (next) {
        case 's': case 't': case 'd': case 'm':
        case 'S': case 'T': case 'D': case 'M':
            return 2; // 's 't 'd 'm (及大写)
        case 'l':
            if (p + 2 < end && (p[2] == 'l' || p[2] == 'L')) return 3; // 'll 'LL
            return 0;
        case 'L':
            if (p + 2 < end && (p[2] == 'l' || p[2] == 'L')) return 3;
            return 0;
        case 'v':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3; // 've 'VE
            return 0;
        case 'V':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3;
            return 0;
        case 'r':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3; // 're 'RE
            return 0;
        case 'R':
            if (p + 2 < end && (p[2] == 'e' || p[2] == 'E')) return 3;
            return 0;
        default:
            return 0;
    }
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    const char* p = text.data();
    const char* end = p + text.size();

    auto emit = [&](const char* from, const char* to) {
        if (from < to) pieces.emplace_back(from, to - from);
    };

    while (p < end) {
        int clen;
        char32_t cp = utf8_decode(p, clen);

        // ===== 规则 1: '(?i:[sdmt]|ll|ve|re) 英文缩写 =====
        if (cp == '\'' && match_contraction(p, end) > 0) {
            int mlen = match_contraction(p, end);
            emit(p, p + mlen);
            p += mlen;
            continue;
        }

        // ===== 规则 2: [^\r\n\p{L}\p{N}]?\p{L}+ =====
        bool is_letter = unicode_is_letter(cp);
        bool is_digit = unicode_is_digit(cp);
        bool is_space = unicode_is_space(cp);
        bool is_newline = unicode_is_newline(cp);

        if (is_letter) {
            const char* start = p;
            p += clen;
            while (p < end) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (unicode_is_letter(ncp)) {
                    p += nlen;
                } else {
                    break;
                }
            }
            emit(start, p);
            continue;
        }

        // 有前导非字母数字非换行的 + 字母
        if (!is_space && !is_newline && !is_digit && !is_letter && p + clen < end) {
            // 向前看：前导字符后面是否紧跟字母
            int nlen;
            char32_t ncp = utf8_decode(p + clen, nlen);
            if (unicode_is_letter(ncp)) {
                const char* start = p;
                p += clen; // 跳过前导字符
                // 消耗字母序列
                while (p < end) {
                    char32_t c2 = utf8_decode(p, nlen);
                    if (unicode_is_letter(c2)) {
                        p += nlen;
                    } else {
                        break;
                    }
                }
                emit(start, p);
                continue;
            }
        }

        // ===== 规则 3: \p{N}{1,3} =====
        if (is_digit) {
            const char* start = p;
            int count = 0;
            while (p < end && count < 3) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (unicode_is_digit(ncp)) {
                    p += nlen;
                    count++;
                } else {
                    break;
                }
            }
            emit(start, p);
            continue;
        }

        // ===== 规则 4:  ?[^\s\p{L}\p{N}]+[\r\n]* =====
        // 可选空格 + 标点/符号 + 可选拖尾换行
        if (!is_space && !is_newline && !is_digit && !is_letter) {
            const char* start = p;
            // 如果前面是空格（作为前导），回退包含它
            // 注意: 到这里说明前面不是字母也不是数字，前面可能是空格
            // 实际上前导空格在前面 "有前导字符 + 字母" 规则中已处理
            // 纯标点/符号序列
            while (p < end) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (!unicode_is_space(ncp) && !unicode_is_letter(ncp) && !unicode_is_digit(ncp)) {
                    p += nlen;
                } else {
                    break;
                }
            }
            // 拖尾换行
            while (p < end && unicode_is_newline(utf8_decode(p, clen))) {
                p += clen;
            }
            emit(start, p);
            continue;
        }

        // 空格作为标点的前导: " ?[^\s\p{L}\p{N}]"
        if (cp == ' ') {
            // 向前看: 空格后面是否跟着标点
            if (p + 1 < end) {
                int nlen;
                char32_t ncp = utf8_decode(p + 1, nlen);
                if (!unicode_is_space(ncp) && !unicode_is_newline(ncp) &&
                    !unicode_is_letter(ncp) && !unicode_is_digit(ncp)) {
                    // 空格 + 标点
                    const char* start = p;
                    p += 1; // 跳过空格
                    while (p < end) {
                        int nnlen;
                        char32_t nncp = utf8_decode(p, nnlen);
                        if (!unicode_is_space(nncp) && !unicode_is_letter(nncp) && !unicode_is_digit(nncp)) {
                            p += nnlen;
                        } else {
                            break;
                        }
                    }
                    // 拖尾换行
                    while (p < end && unicode_is_newline(utf8_decode(p, clen))) {
                        p += clen;
                    }
                    emit(start, p);
                    continue;
                }
            }
        }

        // ===== 规则 5: \s*[\r\n]+ =====
        if (is_space || is_newline) {
            const char* start = p;
            // 消耗前置空白
            while (p < end) {
                int nlen;
                char32_t ncp = utf8_decode(p, nlen);
                if (unicode_is_space(ncp) && !unicode_is_newline(ncp)) {
                    p += nlen;
                } else {
                    break;
                }
            }
            // 如果后面有换行，匹配规则 5
            if (p < end && unicode_is_newline(utf8_decode(p, clen))) {
                while (p < end) {
                    int nlen;
                    char32_t ncp = utf8_decode(p, nlen);
                    if (unicode_is_newline(ncp)) {
                        p += nlen;
                    } else {
                        break;
                    }
                }
                emit(start, p);
                continue;
            }

            // ===== 规则 6/7: \s+(?!\S) 或 \s+ =====
            // 检查是否是拖尾空白（后面没有非空白字符）
            bool trailing = (p >= end) || unicode_is_space(utf8_decode(p, clen));
            p = end; // 已经消耗到末尾或空白末尾
            // 回退: start 到 p 就是空白
            emit(start, p);
            continue;
        }

        // Fallback: 单字符
        p += clen;
        emit(p - clen, p);
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
    if (word.empty()) return {};

    std::vector<Symbol> symbols;
    const char* p = word.data();
    const char* wend = p + word.size();
    while (p < wend) {
        int len = utf8_char_len(*p);
        symbols.push_back({std::string(p, len)});
        p += len;
    }
    if (symbols.empty()) return {};

    if (symbols.size() == 1) {
        auto it = vocab_map_.find(symbols[0].text);
        return (it != vocab_map_.end()) ? std::vector<int32_t>{it->second} : std::vector<int32_t>{};
    }

    // 建立链表
    for (int i = 0; i < static_cast<int>(symbols.size()); ++i) {
        symbols[i].prev = i - 1;
        symbols[i].next = (i + 1 < static_cast<int>(symbols.size())) ? i + 1 : -1;
    }

    // 贪心 BPE: 每轮找 rank 最小的相邻 pair 合并
    while (true) {
        int best_rank = INT32_MAX;
        int best_idx = -1;

        for (int i = 0; i < static_cast<int>(symbols.size()); ++i) {
            if (symbols[i].next < 0) continue;
            int j = symbols[i].next;
            auto it = bpe_ranks_.find({symbols[i].text, symbols[j].text});
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;

        int j = symbols[best_idx].next;
        symbols[best_idx].text += symbols[j].text;
        symbols[best_idx].next = symbols[j].next;
        if (symbols[j].next >= 0)
            symbols[symbols[j].next].prev = best_idx;
        symbols[j].prev = -2; // 标记已删除
    }

    // 沿链表收集结果
    std::vector<int32_t> result;
    for (int i = 0; i < static_cast<int>(symbols.size()); ++i) {
        if (symbols[i].prev == -2) continue; // 已合并
        if (symbols[i].prev != -1 && symbols[i].prev != -2) continue; // 不是 head
        // 从 head 开始遍历
        int cur = i;
        while (cur >= 0) {
            auto it = vocab_map_.find(symbols[cur].text);
            if (it != vocab_map_.end()) result.push_back(it->second);
            cur = symbols[cur].next;
        }
        break; // 只有一个 head
    }
    return result;
}

// ========== encode ==========

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> tokens;

    if (add_bos_ && bos_id_ >= 0)
        tokens.push_back(bos_id_);

    size_t pos = 0;
    while (pos < text.size()) {
        size_t earliest = std::string::npos;
        int32_t matched_id = -1;
        size_t matched_len = 0;

        for (const auto& [tok_text, tok_id] : added_tokens_) {
            size_t found = text.find(tok_text, pos);
            if (found < earliest) {
                earliest = found;
                matched_id = tok_id;
                matched_len = tok_text.size();
            }
        }

        if (earliest == std::string::npos) {
            std::string rest = text.substr(pos);
            auto pieces = pre_tokenize(rest);
            for (auto& piece : pieces) {
                std::string mapped = bytes_to_unicode(piece);
                auto ids = bpe(mapped);
                tokens.insert(tokens.end(), ids.begin(), ids.end());
            }
            break;
        }

        if (earliest > pos) {
            std::string before = text.substr(pos, earliest - pos);
            auto pieces = pre_tokenize(before);
            for (auto& piece : pieces) {
                std::string mapped = bytes_to_unicode(piece);
                auto ids = bpe(mapped);
                tokens.insert(tokens.end(), ids.begin(), ids.end());
            }
        }

        tokens.push_back(matched_id);
        pos = earliest + matched_len;
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
