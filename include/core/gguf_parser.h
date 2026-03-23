// gguf_parser.h
#pragma once
#include <cstdint>
#include <fstream>
#include <print>
#include "3rd/json.hpp"  // nlohmann/json 库
#include "utils/utils.hpp"


using Json = nlohmann::ordered_json;
// 张量信息结构
struct TensorInfo {
    std::string name;
    DataType dtype;
    std::vector<int64_t> dimensions;
    uint64_t offset;  // 绝对文件偏移量
};
// GGUF 头部信息结构
struct GGUFInfo {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    Json metadata;
    std::vector<TensorInfo> tensors_info;

    std::string get_model_architecture() const {
        if (metadata.contains("general.architecture") && metadata["general.architecture"].is_string()) {
            return metadata["general.architecture"].get<std::string>();
        }
        return "unknown";
    }
    std::string get_model_name()const{
        if (metadata.contains("general.name") && metadata["general.name"].is_string()) {
            return metadata["general.name"].get<std::string>();
        }
        return "unknown";
    }
    void print_info() const{
        std::println("=== GGUF 文件信息 ===");
        std::println("gguf:         {}", version);
        std::println("arch:         {}", get_model_architecture());
        std::println("name:         {}", get_model_name());
        std::println("tensor_count: {}", tensor_count);
        std::println("kv_count:     {}", metadata_kv_count);
        // 打印张量信息
        std::println("{:<35} {:<10} {}", "名称", "数据类型", "维度");
        std::println("{:-<35} {:-<10} {:-<20}", "", "", "");
        for(const auto& info:tensors_info){
            std::println("{:<35} {:<10} {}", info.name, data_type_to_string(info.dtype), info.dimensions);
        }
    }
};
// GGUF 解析器类
class GGUFParser {
private:
    std::ifstream file;
public:
    explicit GGUFParser(const std::string& filename);
    ~GGUFParser();
    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;
    GGUFParser(GGUFParser&&) noexcept = default;
    GGUFParser& operator=(GGUFParser&&) noexcept = default;
    GGUFInfo parse();
private:
    uint8_t read_uint8_le();
    uint16_t read_uint16_le();
    uint32_t read_uint32_le();
    uint64_t read_uint64_le();
    float read_float32();
    double read_float64_le();
    std::string read_string();
    Json read_metadata_value(GGUFType type);
    Json parse_metadata(uint64_t kv_count);
    std::vector<TensorInfo> parse_tensors_info(uint64_t tensor_count);
};
