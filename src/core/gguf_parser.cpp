#include "gguf_parser.h"
#include <cassert>
#include <cstddef>

GGUFParser::GGUFParser(const std::string& filename,bool pre_transpose) {
    file_.open(filename, std::ios::binary);
    if (!file_) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    char magic[4];
    if (!file_.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        throw std::runtime_error("Invalid GGUF file: magic number mismatch (expected 'GGUF')");
    }
    this->info_ = parse();
    this->info_.pre_transpose = pre_transpose;
}

GGUFParser::~GGUFParser() {
    if (file_.is_open()) {
        file_.close();
    }
}

void GGUFParser::read_tensor_data(uint64_t tensor_offset, void* dst, size_t size,const Tensor* tensor) {
    uint64_t abs_offset = data_offset_ + tensor_offset;
    file_.seekg(static_cast<std::streamoff>(abs_offset));
    if (!file_) {
        throw std::runtime_error(std::format(
            "GGUFParser: seek to {} (base={}) failed", abs_offset, data_offset_));
    }

    if (info_.pre_transpose) {
        // 先读到临时缓冲区
        std::vector<std::byte> buf(size); // dim=[tensor->dims[1],tensor->dims[0]],需要转置到 [tensor->dims[0],tensor->dims[1]]
        file_.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
        if (!file_) {
            throw std::runtime_error(std::format("GGUFParser: read {} bytes at offset {} failed", size, abs_offset));
        }
        // 判断张量维度数
        int ndim = 0;
        for (int i = 0; i < TENSOR_MAX_DIMS && tensor->dims[i] != 0; ++i) ++ndim;

        if (ndim == 2) {
            size_t esz = data_type_size(tensor->dtype);
            int64_t rows = tensor->dims[1]; // 文件中的行数
            int64_t cols = tensor->dims[0]; // 文件中的列数
            auto* out = static_cast<std::byte*>(dst);
            // 转置 [rows, cols] → [cols, rows]
            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t j = 0; j < cols; ++j) {
                    std::memcpy(out + (j * rows + i) * esz,
                                buf.data() + (i * cols + j) * esz,
                                esz);
                }
            }
        } else {
            std::memcpy(dst, buf.data(), size);
        }
    } else {
        file_.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
        if (!file_) {
            throw std::runtime_error(std::format("GGUFParser: read {} bytes at offset {} failed", size, abs_offset));
        }
    }
}

GGUFInfo GGUFParser::parse() {
    GGUFInfo info;
    info.version = read_uint32_le();
    info.tensor_count = read_uint64_le();
    info.metadata_kv_count = read_uint64_le();

    // 验证版本（当前支持 v3）
    if (info.version != 3) {
        throw std::runtime_error(
            "Unsupported GGUF version: " + std::to_string(info.version) +
            " (only version 3 is supported)"
        );
    }

    // 解析 metadata
    info.metadata = this->parse_metadata(info.metadata_kv_count);

    // 解析张量信息
    info.tensors_info = this->parse_tensors_info(info.tensor_count);


    uint32_t alignment = 32;
    uint64_t raw_offset = static_cast<uint64_t>(file_.tellg());
    if (info_.metadata.contains("general.alignment")) {
        alignment = info_.metadata["general.alignment"].get<uint32_t>();
    }
    data_offset_ = raw_offset + (alignment - raw_offset % alignment) % alignment; // 5951936

    return info;
}
Json GGUFParser::parse_metadata(uint64_t kv_count) {
    Json metadata;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = read_string();
        metadata[key] = read_metadata_value(static_cast<GGUFType>(read_uint32_le()));
    }
    return metadata;
}

std::vector<TensorInfo> GGUFParser::parse_tensors_info(uint64_t tensor_count) {
    std::vector<TensorInfo> tensors;
    tensors.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info;
        info.name = read_string();
        uint32_t dims = read_uint32_le();
        assert(dims == 1 || dims == 2); // 当前版本只支持1/2维张量
        info.dimensions.resize(dims);
        for (uint32_t d = 0; d < dims; d++) {
            if(!this->info_.pre_transpose){ // 目前只对二维权重张量预转置，其他张量保持原维度顺序
                info.dimensions[d] = static_cast<int64_t>(read_uint64_le());
            } else {
                info.dimensions[dims - d -1] = static_cast<int64_t>(read_uint64_le()); // 反转维度顺序，更符合pytorch的行优先存储习惯
            }
        }
        info.dtype = static_cast<DataType>(read_uint32_le());
        info.offset = read_uint64_le();
        tensors.push_back(std::move(info));
    }

    return tensors;
}

uint8_t GGUFParser::read_uint8_le() {
    int ch = file_.get();
    if (file_.eof()) {
        throw std::runtime_error("Unexpected EOF while reading uint8");
    }
    return static_cast<uint8_t>(ch);
}

uint16_t GGUFParser::read_uint16_le() {
    uint8_t bytes[2];
    if (!file_.read(reinterpret_cast<char*>(bytes), 2)) {
        throw std::runtime_error("Failed to read uint16_t from file");
    }
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t GGUFParser::read_uint32_le() {
    uint8_t bytes[4];
    if (!file_.read(reinterpret_cast<char*>(bytes), 4)) {
        throw std::runtime_error("Failed to read uint32_t from file");
    }
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t GGUFParser::read_uint64_le() {
    uint8_t bytes[8];
    if (!file_.read(reinterpret_cast<char*>(bytes), 8)) {
        throw std::runtime_error("Failed to read uint64_t from file");
    }
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    }
    return value;
}

float GGUFParser::read_float32() {
    uint32_t raw = read_uint32_le();
    float f;
    std::memcpy(&f, &raw, sizeof(float));
    return f;
}

double GGUFParser::read_float64_le() {
    uint8_t bytes[8];
    if (!file_.read(reinterpret_cast<char*>(bytes), 8)) {
        throw std::runtime_error("Failed to read float64 from file");
    }
    uint64_t raw = 0;
    for (int i = 0; i < 8; ++i) {
        raw |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    }
    double value;
    std::memcpy(&value, &raw, sizeof(double));
    return value;
}

std::string GGUFParser::read_string() {
    // 字符串格式: [length: uint64][string_data]
    uint64_t len = read_uint64_le();
    std::string str(len, '\0');
    if (!file_.read(str.data(), len)) {
        throw std::runtime_error("Failed to read string from file");
    }
    return str;
}

Json GGUFParser::read_metadata_value(GGUFType type) {
    switch (type) {
        case GGUFType::GGUF_TYPE_UINT8:
            return read_uint8_le();
        case GGUFType::GGUF_TYPE_INT8:
            return static_cast<int8_t>(read_uint8_le());
        case GGUFType::GGUF_TYPE_UINT16:
            return read_uint16_le();
        case GGUFType::GGUF_TYPE_INT16:
            return static_cast<int16_t>(read_uint16_le());
        case GGUFType::GGUF_TYPE_UINT32:
            return read_uint32_le();
        case GGUFType::GGUF_TYPE_INT32:
            return static_cast<int32_t>(read_uint32_le());
        case GGUFType::GGUF_TYPE_UINT64:
            return read_uint64_le();
        case GGUFType::GGUF_TYPE_INT64:
            return static_cast<int64_t>(read_uint64_le());
        case GGUFType::GGUF_TYPE_FLOAT32:
            return read_float32();
        case GGUFType::GGUF_TYPE_FLOAT64:
            return read_float64_le();
        case GGUFType::GGUF_TYPE_BOOL:
            return read_uint8_le() != 0;
        case GGUFType::GGUF_TYPE_STRING:
            return read_string();
        case GGUFType::GGUF_TYPE_ARRAY: {
            // 数组结构: [elem_type: uint32][array_len: uint64][elements...]
            uint32_t elem_type_val = read_uint32_le();
            uint64_t array_len = read_uint64_le();
            Json arr = Json::array();
            for (uint64_t i = 0; i < array_len; ++i) {
                arr.push_back(read_metadata_value(static_cast<GGUFType>(elem_type_val)));
            }
            return arr;
        }
        default:
            throw std::runtime_error(
                "Unsupported GGUF metadata type: " + gguf_type_to_string(type)
            );
    }
}

