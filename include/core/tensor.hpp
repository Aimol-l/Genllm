#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <print>
#include <format>
#include "utils/utils.hpp"

constexpr int32_t TENSOR_MAX_SRC = 10;
constexpr int32_t TENSOR_MAX_DIMS = 4;
constexpr int32_t TENSOR_MAX_OP_PARAMS = 64;

struct Tensor{
    void* data = nullptr;       // 指向实际数据的指针，保证至少 8 字节对齐
    void* backend = nullptr;    // cpu or cuda or vulakn or .....
    size_t device_id = 0;       // 物理设备标识。
    size_t offset = 0;          // data在内存池中的偏移（从基地址算起
    enum DataType dtype;        // fp16,fp32,bf16,.....
    enum TensorType type;       // bias,input
    enum OperationType op_type; // add,sub,mul,div.....
    std::array<int64_t, TENSOR_MAX_DIMS> dims = {0,0,0,0};      // 各维度大小,默认都是0
    std::array<uint64_t, TENSOR_MAX_DIMS> strides = {0,0,0,0};  // 各维度字节跨度
    std::array<Tensor*, TENSOR_MAX_SRC> src;                    // 源 tensor（用于计算图）
    std::array<float, TENSOR_MAX_OP_PARAMS/sizeof(float)> op_params;  // 操作参数

    std::string name;
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