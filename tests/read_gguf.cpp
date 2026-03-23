#include <iostream>
#include <fstream>

#include "gguf_parser.h"

int main() {
    try {
        GGUFParser parser("models/Qwen3-0.6B-BF16.gguf");

        // 解析GGUF文件
        GGUFInfo info = parser.parse();

        // 打印基本信息
        std::println("=== GGUF 文件信息 ===");
        std::println("版本: {}", info.version);
        std::println("张量数量: {}", info.tensor_count);
        std::println("元数据键值对数量: {}", info.metadata_kv_count);

        // 保存元数据到文件
        std::ofstream meta_file("metadata.json");
        if (meta_file.is_open()) {
            meta_file << info.metadata.dump(4);
            meta_file.close();
            std::println("\n元数据已保存到: metadata.json");
        } else {
            std::println(stderr, "警告: 无法创建元数据文件");
        }

    } catch (const std::exception& e) {
        std::println(stderr, "错误: {}", e.what());
        return 1;
    }
    return 0;
}