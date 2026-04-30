#include <stdexcept>
#include "tensor.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cuda/arithmetic.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

template<typename T>
__global__ void add_kernel(const T* __restrict__ input1,const T* __restrict__ input2,T* __restrict__ output, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        output[glob_id] = input1[glob_id] + input2[glob_id];
    }
}

void AddImpl<Device::CUDA>::execute(Tensor* t) { 
    const Tensor* x = t->src[0];

    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in1 = static_cast<const __half*>(x->data);
            const __half* in2 = static_cast<const __half*>(t->src[1]->data);
            add_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in1 = static_cast<const __nv_bfloat16*>(x->data);
            const __nv_bfloat16* in2 = static_cast<const __nv_bfloat16*>(t->src[1]->data);
            add_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,float>){
            float*      out = static_cast<float*>(t->data);
            const float* in1 = static_cast<const float*>(x->data);
            const float* in2 = static_cast<const float*>(t->src[1]->data);
            add_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else{
            throw std::runtime_error("cuda::add not implemented");
        }
    }); 
}
template<typename T>
__global__ void sub_kernel(const T* __restrict__ input1,const T* __restrict__ input2,T* __restrict__ output, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        output[glob_id] = input1[glob_id] - input2[glob_id];
    }
}
void SubImpl<Device::CUDA>::execute(Tensor* t){
    const Tensor* x = t->src[0];

    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in1 = static_cast<const __half*>(x->data);
            const __half* in2 = static_cast<const __half*>(t->src[1]->data);
            sub_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in1 = static_cast<const __nv_bfloat16*>(x->data);
            const __nv_bfloat16* in2 = static_cast<const __nv_bfloat16*>(t->src[1]->data);
            sub_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,float>){
            float*      out = static_cast<float*>(t->data);
            const float* in1 = static_cast<const float*>(x->data);
            const float* in2 = static_cast<const float*>(t->src[1]->data);
            sub_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else{
            throw std::runtime_error("cuda::sub not implemented");
        }
    }); 
}

template<typename T>
__global__ void mul_kernel(const T* __restrict__ input1,const T* __restrict__ input2,T* __restrict__ output, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        output[glob_id] = input1[glob_id] * input2[glob_id];
    }
}
void MulImpl<Device::CUDA>::execute(Tensor* t){
    const Tensor* x = t->src[0];

    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in1 = static_cast<const __half*>(x->data);
            const __half* in2 = static_cast<const __half*>(t->src[1]->data);
            mul_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in1 = static_cast<const __nv_bfloat16*>(x->data);
            const __nv_bfloat16* in2 = static_cast<const __nv_bfloat16*>(t->src[1]->data);
            mul_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,float>){
            float*      out = static_cast<float*>(t->data);
            const float* in1 = static_cast<const float*>(x->data);
            const float* in2 = static_cast<const float*>(t->src[1]->data);
            mul_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else{
            throw std::runtime_error("cuda::mul not implemented");
        }
    }); 
}

template<typename T>
__global__ void div_kernel(const T* __restrict__ input1,const T* __restrict__ input2,T* __restrict__ output, size_t size) {
    size_t glob_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (glob_id < size) {
        output[glob_id] = input1[glob_id] / input2[glob_id];
    }
}
void DivImpl<Device::CUDA>::execute(Tensor* t){
    const Tensor* x = t->src[0];

    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in1 = static_cast<const __half*>(x->data);
            const __half* in2 = static_cast<const __half*>(t->src[1]->data);
            div_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in1 = static_cast<const __nv_bfloat16*>(x->data);
            const __nv_bfloat16* in2 = static_cast<const __nv_bfloat16*>(t->src[1]->data);
            div_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else if constexpr(std::is_same_v<T,float>){
            float*      out = static_cast<float*>(t->data);
            const float* in1 = static_cast<const float*>(x->data);
            const float* in2 = static_cast<const float*>(t->src[1]->data);
            div_kernel<<<blocks, threads>>>(in1, in2, out, numel);
        }else{
            throw std::runtime_error("cuda::div not implemented");
        }
    }); 

}
template struct AddImpl<Device::CUDA>;
template struct SubImpl<Device::CUDA>;
template struct MulImpl<Device::CUDA>;
template struct DivImpl<Device::CUDA>;
}
