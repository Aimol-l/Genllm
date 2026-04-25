#include <cmath>
#include <stdexcept>
#include "tensor.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cuda/activation.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"


namespace ops {
template <typename T> // output =  x / (1 + exp(-x))
__global__ void silu_kernel(const T* __restrict__ input, T* __restrict__ output, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        T x = input[glob_id];
        output[glob_id] = x / T(1 + expf(-x));
    }
}

void SiluImpl<Device::CUDA>::execute(Tensor* t) {
    const Tensor* x = t->src[0];

    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in = static_cast<const __half*>(x->data);
            silu_kernel<<<blocks, threads>>>(in, out, numel);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in = static_cast<const __nv_bfloat16*>(x->data);
            silu_kernel<<<blocks, threads>>>(in, out, numel);
        }else{
            throw std::runtime_error("cuda::silu not implemented");
        }
    });
}

template <typename T> // output =  x / (1 + exp(-x))
__global__ void gelu_kernel(const T* __restrict__ input, T* __restrict__ output, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;

    constexpr float inv_sqrt2 = 0.7071067811865475f;

    if (glob_id < size) {
        float fx = float(input[glob_id]);
        output[glob_id] = T(fx * 0.5f * (1.0f + std::erf(fx * inv_sqrt2)));
    }
}
void GeluImpl<Device::CUDA>::execute(Tensor* t) {
    const Tensor* x = t->src[0];

    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in = static_cast<const __half*>(x->data);
            gelu_kernel<<<blocks, threads>>>(in, out, numel);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in = static_cast<const __nv_bfloat16*>(x->data);
            gelu_kernel<<<blocks, threads>>>(in, out, numel);
        }else{
            throw std::runtime_error("cuda::gelu not implemented");
        }
    });
}

template <typename T>
__global__ void relu_kernel(const T* __restrict__ input, T* __restrict__ output, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        T x = input[glob_id];
        output[glob_id] = x > T(0) ? x : T(0);
    }
}
void ReluImpl<Device::CUDA>::execute(Tensor* t) {
    const Tensor* x = t->src[0];

    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in = static_cast<const __half*>(x->data);
            relu_kernel<<<blocks, threads>>>(in, out, numel);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in = static_cast<const __nv_bfloat16*>(x->data);
            relu_kernel<<<blocks, threads>>>(in, out, numel);
        }else{
            throw std::runtime_error("cuda::relu not implemented");
        }
    });
}

template struct SiluImpl<Device::CUDA>;
template struct GeluImpl<Device::CUDA>;
template struct ReluImpl<Device::CUDA>;
}
