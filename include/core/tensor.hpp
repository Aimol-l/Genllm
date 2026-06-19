#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include "utils/utils.hpp"

constexpr int32_t TENSOR_MAX_SRC = 8;
constexpr int32_t TENSOR_MAX_DIMS = 4;
constexpr int32_t TENSOR_MAX_OP_PARAMS = 4;

struct Tensor{
    // =====cache line 0(8+8+8*8 = 80)=====
    void* data = nullptr;       // CPU/CUDA: 指向实际数据的指针；Vulkan: nullptr（使用 device_handle）
    size_t offset = 0;          // data 在内存池中的偏移（从基地址算起）
    std::array<Tensor*, TENSOR_MAX_SRC> src = {nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr};// 源 tensor（用于计算图）

    // =====cache line 1(4+8+1+1+1+1+32+16)=====
    int layer_id = -1;
    size_t device_handle = 0;   
    enum Device device = Device::CPU;                            
    enum DataType dtype = DataType::GGML_TYPE_F32;               
    enum TensorType type = TensorType::TENSOR_TYPE_UNKNOWN;   
    enum OperationType op_type = OperationType::OP_TYPE_NONE;
    std::array<int64_t, TENSOR_MAX_DIMS> dims = {0,0,0,0};
    std::array<float, TENSOR_MAX_OP_PARAMS> op_params = {0,0,0,0};     // 操作参数


    std::array<uint64_t, TENSOR_MAX_DIMS> strides = {0,0,0,0};         // 元素跨度
    std::string name;

    // ========== 工具方法 ==========
    size_t num_elements() const {
        size_t total = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && dims[i] != 0; ++i) {
            total *= std::abs(dims[i]);
        }
        return total;
    }
    size_t bytes() const {
        return num_elements() * data_type_size(dtype);
    }
    size_t bytes_at(int64_t resolve) const {
        size_t total = 1;
        for (int i = 0; i < TENSOR_MAX_DIMS && dims[i] != 0; ++i) {
            total *= (dims[i] < 0 ? resolve : dims[i]);
        }
        return total * data_type_size(dtype);
    }
    bool is_computed() const {
        return op_type != OperationType::OP_TYPE_NONE && type != TensorType::TENSOR_TYPE_WEIGHT && type != TensorType::TENSOR_TYPE_INPUT;
    }
};
