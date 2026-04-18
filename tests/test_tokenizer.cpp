#include "tokenizer.h"
#include "gguf_parser.h"
#include <print>

int main() {
    GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");
    Tokenizer tokenizer = Tokenizer::from_gguf(parser);

    int pass = 0, fail = 0;

    auto test = [&](const std::string& text) {
        auto ids = tokenizer.encode(text);
        auto dec = tokenizer.decode(ids);
        bool ok = (dec == text);
        std::print("[{}] \"{}\" → {} tokens [", ok ? "OK" : "FAIL", text, ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) std::print(", ");
            std::print("{}", ids[i]);
        }
        std::println("] → \"{}\" {}", dec, ok ? "" : "(mismatch!)");
        if (ok) ++pass; else ++fail;
    };

    test("Once upon a time");
    test("Hello, world!");
    test("The quick brown fox jumps over the lazy dog.");
    test("  multiple   spaces  ");
    test("numbers 12345 and symbols !@#$%");
    test("");
    test("a");
    test("I'm can't won't they're we've");
    test("line1\nline2\nline3");
    test("Hello\n  world!");

    std::println("\n{} passed, {} failed", pass, fail);
    return fail > 0 ? 1 : 0;
}
