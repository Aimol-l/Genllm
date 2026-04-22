#pragma once
#include <stdexcept>
#include <format>
#include "utils/utils.hpp"
#include "utils/float16.hpp"
#include "utils/bfloat16.hpp"
#include "utils/utils.hpp"

namespace dtype {

// ==================== DataType -> C++ type 映射 ====================

template<DataType D> struct Traits;

template<> struct Traits<DataType::GGML_TYPE_F32>  { using T = float;      static constexpr DataType value = DataType::GGML_TYPE_F32;  };
template<> struct Traits<DataType::GGML_TYPE_F16>  { using T = float16_t;  static constexpr DataType value = DataType::GGML_TYPE_F16;  };
template<> struct Traits<DataType::GGML_TYPE_BF16> { using T = bfloat16_t; static constexpr DataType value = DataType::GGML_TYPE_BF16; };

template<DataType D>
using type_t = typename Traits<D>::T;

// ==================== 编译时类型转换 ====================

template<DataType D>
inline float to_f32(type_t<D> v);

template<> inline float to_f32<DataType::GGML_TYPE_F32>(float v)       { return v; }
template<> inline float to_f32<DataType::GGML_TYPE_F16>(float16_t v)   { return static_cast<float>(v); }
template<> inline float to_f32<DataType::GGML_TYPE_BF16>(bfloat16_t v) { return static_cast<float>(v); }

template<DataType D>
inline type_t<D> from_f32(float f);

template<> inline float      from_f32<DataType::GGML_TYPE_F32>(float f)  { return f; }
template<> inline float16_t  from_f32<DataType::GGML_TYPE_F16>(float f)  { return float16_t(f); }
template<> inline bfloat16_t from_f32<DataType::GGML_TYPE_BF16>(float f) { return bfloat16_t(f); }

// ==================== 运行时分发器 ====================

// 分发浮点类型 (F32/F16/BF16)，Fn 是一个接受编译时 DataType 的可调用对象
template<typename Fn>
void dispatch(DataType dtype, Fn&& fn) {
    switch (dtype) {
        case DataType::GGML_TYPE_F32:  fn.template operator()<DataType::GGML_TYPE_F32>();  break;
        case DataType::GGML_TYPE_F16:  fn.template operator()<DataType::GGML_TYPE_F16>();  break;
        case DataType::GGML_TYPE_BF16: fn.template operator()<DataType::GGML_TYPE_BF16>(); break;
        default: throw std::runtime_error(std::format("dtype::dispatch: unsupported dtype {}", data_type_to_string(dtype)));
    }
}

// 运行时转换: 任意浮点 dtype 的值 -> float
inline float to_f32_rt(DataType dtype, const void* ptr) {
    switch (dtype) {
        case DataType::GGML_TYPE_F32:  return *static_cast<const float*>(ptr);
        case DataType::GGML_TYPE_F16:  return static_cast<float>(*static_cast<const float16_t*>(ptr));
        case DataType::GGML_TYPE_BF16: return static_cast<float>(*static_cast<const bfloat16_t*>(ptr));
        default: throw std::runtime_error(std::format("dtype::to_f32_rt: unsupported dtype {}", data_type_to_string(dtype)));
    }
}

// 运行时转换: float -> 任意浮点 dtype 的值
inline void from_f32_rt(DataType dtype, float f, void* ptr) {
    switch (dtype) {
        case DataType::GGML_TYPE_F32:  *static_cast<float*>(ptr)      = f;             break;
        case DataType::GGML_TYPE_F16:  *static_cast<float16_t*>(ptr)  = float16_t(f);  break;
        case DataType::GGML_TYPE_BF16: *static_cast<bfloat16_t*>(ptr) = bfloat16_t(f); break;
        default: throw std::runtime_error(std::format("dtype::from_f32_rt: unsupported dtype {}", data_type_to_string(dtype)));
    }
}

} // namespace dtype

namespace device {

// Device 分发：运行时 Device → 编译时模板参数
template<typename Fn>
void dispatchOp(Device device, Fn&& fn) {
    switch (device) {
        case Device::CPU: fn.template operator()<Device::CPU>(); break;
    #ifdef BACKEND_CUDA
        case Device::CUDA: fn.template operator()<Device::CUDA>(); break;
    #endif
        default: throw std::runtime_error(std::format("device::dispatchOp: unsupported device {}", device_to_string(device)));
    }
}

};