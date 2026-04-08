#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include "utils/utils.hpp"

constexpr int32_t TENSOR_MAX_SRC = 5;
constexpr int32_t TENSOR_MAX_DIMS = 4;
constexpr int32_t TENSOR_MAX_OP_PARAMS = 4;

struct Tensor{
    void* data = nullptr;       // 指向实际数据的指针，保证至少 8 字节对齐
    size_t offset = 0;          // data在内存池中的偏移（从基地址算起
    int layer_id = -1;
    enum Device device;        // 逻辑设备（CPU/CUDA/SYCL/VULKAN）
    enum DataType dtype;        // fp16,fp32,bf16,.....
    enum TensorType type;       // bias,input
    enum OperationType op_type; // add,sub,mul,div.....
    std::array<int64_t, TENSOR_MAX_DIMS> dims = {0,0,0,0};      // 各维度大小,默认都是0
    std::array<uint64_t, TENSOR_MAX_DIMS> strides = {0,0,0,0};  // 各维度字节跨度
    std::array<Tensor*, TENSOR_MAX_SRC> src;                    // 源 tensor（用于计算图）
    std::array<float, TENSOR_MAX_OP_PARAMS> op_params;          // 操作参数
    std::string name;
    // ========== 工具方法 ==========
    size_t num_elements() const {
        size_t total = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && dims[i] > 0; ++i) {
            total *= dims[i];
        }
        return total;
    }
    size_t bytes() const {
        return num_elements() * data_type_size(dtype);
    }
    bool is_computed() const {
        return op_type != OperationType::OP_TYPE_NONE 
            && type != TensorType::TENSOR_TYPE_WEIGHT 
            && type != TensorType::TENSOR_TYPE_INPUT;
    }
};
// 辅助函数：获取张量维度字符串
inline std::string tensor_dims_to_string(const Tensor* t) {
    if (!t) return "[]";
    std::string s = "[";
    bool first = true;
    for (const auto& dim : t->dims) {
        if (dim == 0 && !first) break;
        if (!first) s += ", ";
        s += std::to_string(dim);
        first = false;
    }
    s += "]";
    return s;
}
