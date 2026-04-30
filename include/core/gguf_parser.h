// gguf_parser.h
#pragma once
#include <cstdint>
#include <fstream>
#include "3rd/json.hpp"  // nlohmann/json 库
#include "tensor.hpp"
#include "utils/utils.hpp"

using Json = nlohmann::ordered_json;
// 张量信息结构
struct TensorInfo {
    bool transpose  = false;
    DataType dtype  = DataType::GGML_TYPE_F32;
    uint64_t offset = 0;                 // 绝对文件偏移量,gguf上的
    std::string name = "unknown";
    std::vector<int64_t> dimensions;     //transpose后维度，底层数据也要在加载时转置
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
    uint32_t offset;    // 张量起始地址偏移
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    std::vector<TensorInfo> tensors_info;

    void print_info() const;
    std::string get_model_name() const;
    std::string get_model_architecture() const;

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
    void read_tensor_data(uint64_t tensor_offset, void* dst, size_t size,const Tensor* tensor);
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
