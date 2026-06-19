#include <cmath>
#include "utils/float16.hpp"
#include "utils/bfloat16.hpp"
#include "utils/utils.hpp"
#include "backend/cpu/activation.h"

namespace ops {

    template<typename T>
    void silu_(const T* __restrict__ input, T* __restrict__ output, size_t len){
        for (size_t i = 0; i < len; ++i) {
            float fx = float(input[i]);
            output[i] = T(fx / (1.0f + std::exp(-fx)));
        }
    }
    void SiluImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        size_t len = out->num_elements();
        switch (out->dtype) {
            case DataType::GGML_TYPE_F32:
                silu_<float>(static_cast<const float*>(x->data),static_cast<float*>(out->data),len); break;
            case DataType::GGML_TYPE_F16: 
                silu_<float16_t>(static_cast<const float16_t*>(x->data),static_cast<float16_t*>(out->data),len); break;
            case DataType::GGML_TYPE_BF16:
                silu_<bfloat16_t>(static_cast<const bfloat16_t*>(x->data),static_cast<bfloat16_t*>(out->data),len); break;
            default: 
                throw std::runtime_error(std::format("silu: unsupported dtype {}", data_type_to_string(out->dtype)));
        }
    }
    template<typename T>
    void gelu_(const T* __restrict__ input, T* __restrict__ output, size_t len){
        constexpr float inv_sqrt2 = 0.7071067811865475f;
        for (size_t i = 0; i < len; ++i) {
            float fx = float(input[i]);
            output[i] = T(fx * 0.5f * (1.0f + std::erf(fx * inv_sqrt2)));
        }
    }
    void GeluImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        size_t len = out->num_elements();
        switch (out->dtype) {
            case DataType::GGML_TYPE_F32:
                gelu_<float>(static_cast<const float*>(x->data),static_cast<float*>(out->data),len); break;
            case DataType::GGML_TYPE_F16: 
                gelu_<float16_t>(static_cast<const float16_t*>(x->data),static_cast<float16_t*>(out->data),len); break;
            case DataType::GGML_TYPE_BF16:
                gelu_<bfloat16_t>(static_cast<const bfloat16_t*>(x->data),static_cast<bfloat16_t*>(out->data),len); break;
            default: 
                throw std::runtime_error(std::format("gelu_: unsupported dtype {}", data_type_to_string(out->dtype)));
        }
    }
    template<typename T>
    void relu_(const T* __restrict__ input, T* __restrict__ output, size_t len){
        for (size_t i = 0; i < len; ++i) {
           output[i] = input[i] > T(0) ? input[i] : T(0);
        }
    }
    void ReluImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        size_t len = out->num_elements();
        switch (out->dtype) {
            case DataType::GGML_TYPE_F32:
                relu_<float>(static_cast<const float*>(x->data),static_cast<float*>(out->data),len); break;
            case DataType::GGML_TYPE_F16: 
                relu_<float16_t>(static_cast<const float16_t*>(x->data),static_cast<float16_t*>(out->data),len); break;
            case DataType::GGML_TYPE_BF16:
                relu_<bfloat16_t>(static_cast<const bfloat16_t*>(x->data),static_cast<bfloat16_t*>(out->data),len); break;
            default: 
                throw std::runtime_error(std::format("relu_: unsupported dtype {}", data_type_to_string(out->dtype)));
        }
    }
    template<typename T>
    void sigmoid_(const T* __restrict__ input, T* __restrict__ output, size_t len){
        for (size_t i = 0; i < len; ++i) {
          output[i] = T(1.0f / (1.0f + std::exp(-float(input[i]))));
        }
    }
    void SigmoidImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        size_t len = out->num_elements();
        switch (out->dtype) {
            case DataType::GGML_TYPE_F32:
                sigmoid_<float>(static_cast<const float*>(x->data),static_cast<float*>(out->data),len); break;
            case DataType::GGML_TYPE_F16: 
                sigmoid_<float16_t>(static_cast<const float16_t*>(x->data),static_cast<float16_t*>(out->data),len); break;
            case DataType::GGML_TYPE_BF16:
                sigmoid_<bfloat16_t>(static_cast<const bfloat16_t*>(x->data),static_cast<bfloat16_t*>(out->data),len); break;
            default: 
                throw std::runtime_error(std::format("sigmoid: unsupported dtype {}", data_type_to_string(out->dtype)));
        }
    }

template struct SiluImpl<Device::CPU>;
template struct GeluImpl<Device::CPU>;
template struct ReluImpl<Device::CPU>;
template struct SigmoidImpl<Device::CPU>;

}
