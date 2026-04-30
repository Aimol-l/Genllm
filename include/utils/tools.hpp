#pragma once
#include "tensor.hpp"
#include "utils/bfloat16.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <vector>
#include <memory>
#include <concepts>
#include <format>
#include <print>
#include <cstring>
#ifdef BACKEND_CUDA
#include <cuda_runtime.h>
#endif

// ANSI 颜色代码
constexpr const char* COLOR_RESET  = "\033[0m";
constexpr const char* COLOR_RED    = "\033[31m";
constexpr const char* COLOR_YELLOW = "\033[33m";
constexpr const char* COLOR_CYAN   = "\033[36m";

// 日志基础宏（内部使用）
#define LOG_BASE(level, color, msg) do { \
    auto now = std::chrono::system_clock::now(); \
    std::time_t t = std::chrono::system_clock::to_time_t(now); \
    std::tm tm_info; \
    localtime_r(&t, &tm_info); \
    std::cerr << color << "[" << level << "] " \
              << std::put_time(&tm_info, "%Y-%m-%d %H:%M:%S") \
              << " [" << __FILE__ << ":" << __LINE__ << "]" \
              << " (" << __func__ << "): " << msg << COLOR_RESET << std::endl; \
} while (0)

// 公开使用的日志宏
#define LOG_INFO(msg)  LOG_BASE("INFO",  COLOR_CYAN,   msg)
#define LOG_WARN(msg)  LOG_BASE("WARN",  COLOR_YELLOW, msg)
#define LOG_ERROR(msg) LOG_BASE("ERROR", COLOR_RED,    msg)

#define RUNNING_TIME(expr)                                                          \
    do {                                                                            \
        auto start = std::chrono::steady_clock::now();                              \
        expr;                                                                       \
        auto end = std::chrono::steady_clock::now();                                \
        std::chrono::duration<double, std::milli> duration_ms = end - start;        \
        double ms = duration_ms.count();                                            \
        std::cout << std::fixed << std::setprecision(3);                            \
        if (ms < 1000.0) {                                                          \
            std::cout << "Execution time: " << ms << " ms" << std::endl;             \
        } else if (ms < 60.0 * 1000.0) {                                            \
            std::cout << "Execution time: " << ms / 1000.0 << " s" << std::endl;     \
        } else if (ms < 60.0 * 60.0 * 1000.0) {                                     \
            std::cout << "Execution time: " << ms / (60.0 * 1000.0) << " min" << std::endl; \
        } else {                                                                    \
            std::cout << "Execution time: " << ms / (60.0 * 60.0 * 1000.0) << " h" << std::endl; \
        }                                                                           \
    } while (0)


namespace ops{

    // softmax for 1D const span, returns float probabilities
    template <typename T> requires std::same_as<std::remove_cv_t<T>, bfloat16_t> || std::same_as<std::remove_cv_t<T>, float>
    inline std::vector<float> Softmax(std::span<const T> vec, float temperature = 1.0f) {
        if (vec.empty()) return {};

        float inv_temp = 1.0f / temperature;

        // find max in float domain
        float max_val = static_cast<float>(vec[0]);
        for (size_t i = 1; i < vec.size(); ++i)
            max_val = std::max(max_val, static_cast<float>(vec[i]));

        // exp and sum
        std::vector<float> probs(vec.size());
        float sum = 0.0f;
        for (size_t i = 0; i < vec.size(); ++i) {
            probs[i] = std::exp((static_cast<float>(vec[i]) - max_val) * inv_temp);
            sum += probs[i];
        }

        // normalize, clamp to avoid zero probability
        for (auto& p : probs)
            p = std::max(p / sum, 1e-6f);

        return probs;
    }
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
    inline void println(const Tensor* t) {
        if (!t) {
            std::println("Tensor(nullptr)");
            return;
        }
        if (!t->data) {
            std::println(" [data is null]");
            return;
        }
        // Determine ndim and shape
        int ndim = 0;
        int64_t shape[TENSOR_MAX_DIMS];
        for (int i = 0; i < TENSOR_MAX_DIMS && t->dims[i] != 0; ++i)
            shape[ndim++] = t->dims[i];

        if (ndim == 0) {
            std::println(" []");
            return;
        }
        // Check quantized types
        auto is_quantized = [](DataType dt) {
            return dt == DataType::GGML_TYPE_Q4_0  || dt == DataType::GGML_TYPE_Q4_1  ||
                   dt == DataType::GGML_TYPE_Q5_0  || dt == DataType::GGML_TYPE_Q5_1  ||
                   dt == DataType::GGML_TYPE_Q8_0  || dt == DataType::GGML_TYPE_Q8_1  ||
                   dt == DataType::GGML_TYPE_Q2_K  || dt == DataType::GGML_TYPE_Q3_K  ||
                   dt == DataType::GGML_TYPE_Q4_K  || dt == DataType::GGML_TYPE_Q5_K  ||
                   dt == DataType::GGML_TYPE_Q6_K  || dt == DataType::GGML_TYPE_Q8_K  ||
                   dt == DataType::GGML_TYPE_IQ2_XXS || dt == DataType::GGML_TYPE_IQ2_XS ||
                   dt == DataType::GGML_TYPE_IQ3_XXS || dt == DataType::GGML_TYPE_IQ1_S  ||
                   dt == DataType::GGML_TYPE_IQ4_NL  || dt == DataType::GGML_TYPE_IQ3_S  ||
                   dt == DataType::GGML_TYPE_IQ2_S   || dt == DataType::GGML_TYPE_IQ4_XS  ||
                   dt == DataType::GGML_TYPE_IQ1_M   || dt == DataType::GGML_TYPE_TQ1_0  ||
                   dt == DataType::GGML_TYPE_TQ2_0   || dt == DataType::GGML_TYPE_MXFP4;
        };
        if (is_quantized(t->dtype)) {
            std::println(" [quantized, bytes={}]", t->bytes());
            return;
        }

        // Copy device tensor data to a temporary host buffer for printing
        const void* host_data = t->data;
        std::unique_ptr<char[]> device_copy;
        if (t->device != Device::CPU) {
            size_t nbytes = t->bytes();
            device_copy = std::make_unique<char[]>(nbytes);
            switch (t->device) {
            case Device::CUDA:
    #ifdef BACKEND_CUDA
                cudaMemcpy(device_copy.get(), t->data, nbytes, cudaMemcpyDeviceToHost);
    #endif
                break;
            case Device::SYCL:
            case Device::VULKAN:
                break;
            default:
                break;
            }
            host_data = device_copy.get();
        }

        // Read element at flat index as double
        auto read_val = [t, host_data](size_t i) -> double {
            switch (t->dtype) {
            case DataType::GGML_TYPE_F32:
                return static_cast<const float*>(host_data)[i];
            case DataType::GGML_TYPE_F64:
                return static_cast<const double*>(host_data)[i];
            case DataType::GGML_TYPE_F16: {
                uint16_t h = static_cast<const uint16_t*>(host_data)[i];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                if (exp == 0 && mant == 0) return sign ? -0.0 : 0.0;
                if (exp == 0) return std::ldexp(mant / 1024.0, -14) * (sign ? -1.0 : 1.0);
                if (exp == 31) return mant == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
                return std::ldexp(1.0 + mant / 1024.0, exp - 15) * (sign ? -1.0 : 1.0);
            }
            case DataType::GGML_TYPE_BF16: {
                uint16_t h = static_cast<const uint16_t*>(host_data)[i];
                uint32_t bits = static_cast<uint32_t>(h) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                return f;
            }
            case DataType::GGML_TYPE_I8:  return static_cast<const int8_t*      >(host_data)[i];
            case DataType::GGML_TYPE_I16: return static_cast<const int16_t*     >(host_data)[i];
            case DataType::GGML_TYPE_I32: return static_cast<const int32_t*     >(host_data)[i];
            case DataType::GGML_TYPE_I64: return static_cast<const int64_t*     >(host_data)[i];
            default: return 0.0;
            }
        };
        if (is_quantized(t->dtype)) {
            std::println(" [quantized, bytes={}]", t->bytes());
            return;
        }

        bool is_int = (t->dtype >= DataType::GGML_TYPE_I8 && t->dtype <= DataType::GGML_TYPE_I64);

        // Precompute flat strides
        size_t dim_stride[TENSOR_MAX_DIMS];
        dim_stride[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d)
            dim_stride[d] = dim_stride[d + 1] * static_cast<size_t>(shape[d + 1]);

        constexpr size_t EDGE = 3;
        constexpr size_t MAX_PER_DIM = 7;

        // Y-combinator style recursive printer
        auto print_dim = [&](auto&& self, int dim, size_t offset, std::string indent) -> void {
            int64_t n = shape[dim];
            bool omit = n > static_cast<int64_t>(MAX_PER_DIM);
            size_t show = omit ? EDGE : static_cast<size_t>(n);

            if (dim == ndim - 1) {
                // Innermost dimension
                std::print("{}[", indent);
                for (size_t i = 0; i < show; ++i) {
                    if (is_int) std::print("{}", static_cast<int64_t>(read_val(offset + i)));
                    else        std::print("{:.4f}", read_val(offset + i));
                    if (i != show - 1) std::print(", ");
                }
                if (omit) {
                    std::print(", ..., ");
                    for (size_t i = static_cast<size_t>(n) - EDGE; i < static_cast<size_t>(n); ++i) {
                        if (is_int) std::print("{}", static_cast<int64_t>(read_val(offset + i)));
                        else        std::print("{:.4f}", read_val(offset + i));
                        if (i != static_cast<size_t>(n) - 1) std::print(", ");
                    }
                }
                std::print("]");
            } else {
                // Outer dimensions
                std::print("{}[\n", indent);
                for (size_t i = 0; i < show; ++i) {
                    self(self, dim + 1, offset + i * dim_stride[dim], indent + "  ");
                    std::print(",\n");
                }
                if (omit) {
                    std::print("{}  ....\n", indent);
                    for (size_t i = static_cast<size_t>(n) - EDGE; i < static_cast<size_t>(n); ++i) {
                        self(self, dim + 1, offset + i * dim_stride[dim], indent + "  ");
                        if (i != static_cast<size_t>(n) - 1) std::print(",\n");
                        else                              std::print("\n");
                    }
                }
                std::print("{}]", indent);
            }
        };
        print_dim(print_dim, 0, 0, "");
        std::println("\nTensor name= \"{}\" | shape={} | dtype={} | device={}",
            t->name,
            ops::tensor_dims_to_string(t),
            data_type_to_string(t->dtype),
            device_to_string(t->device)
        );
    }
}