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
    bool transpose = false;
    DataType dtype;
    uint64_t offset;                 // 绝对文件偏移量,gguf上的
    std::string name;
    std::vector<int64_t> dimensions; //transpose后维度，底层数据也要在加载时转置
    size_t bytes() const {
        size_t elem_size = data_type_size(dtype);
        size_t num_elems = 1;
        for (int64_t dim : dimensions) {
            num_elems *= dim;
        }
        return elem_size * num_elems;
    }
};
// GGUF 头部信息结构
struct GGUFInfo {
    Json metadata;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
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
        std::println("{:-<10}GGUF 文件信息{:-<10}", "", "");
        std::println("gguf version:             {}", version);
        std::println("model arch:               {}", get_model_architecture());
        std::println("model name:               {}", get_model_name());
        std::println("kv_count:                 {}", metadata_kv_count);
        std::println("tensor_count:             {}", tensor_count);
        // 打印张量信息
        std::println("{:-<26} {:-<8} {:-<14}", "", "", "");
        std::println("{:<26} {:<8} {}", "名称", "数据类型", "维度");
        std::println("{:-<26} {:-<8} {:-<14}", "", "", "");
        for(const auto& info:tensors_info){
            std::println("{:<26} {:<8} {}", info.name, data_type_to_string(info.dtype), info.dimensions);
        }
        std::println("{:-<26} {:-<8} {:-<14}", "", "", "");
    }
};
// GGUF 解析器类
class GGUFParser {
private:
    std::ifstream file_;
    GGUFInfo info_;
    uint64_t data_offset_ = 0;
public:
    explicit GGUFParser(const std::string& filename);
    ~GGUFParser();
    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;
    GGUFParser(GGUFParser&&) noexcept = default;
    GGUFParser& operator=(const GGUFParser&&) noexcept = delete;
    GGUFInfo& info() { return info_; }
    [[nodiscard]] uint64_t data_offset() const { return data_offset_; }
    void read_tensor_data(uint64_t tensor_offset, void* dst, size_t size);
private:
    GGUFInfo parse();
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
