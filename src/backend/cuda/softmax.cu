#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "backend/cuda/softmax.h"

#include "utils/utils.hpp"
#include "utils/dtype_traits.hpp"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

template <typename T>
__global__ void softmax_kernel(const T* __restrict__ input, T* __restrict__ output, size_t outer_dim, size_t axis_dim, size_t inner_dim) { 
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = outer_dim * inner_dim;
    if (idx >= total) return;
    // 计算索引
    size_t outer_idx = idx / inner_dim;
    size_t inner_idx = idx % inner_dim;
    const size_t base_offset = outer_idx * axis_dim * inner_dim + inner_idx;

    float max_val = -INFINITY; 
    for (size_t i = 0; i < axis_dim; ++i) {
        float val = static_cast<float>(input[base_offset + i * inner_dim]);
        max_val = fmaxf(val, max_val);
    }
    float sum = 0.0f;
    for (size_t i = 0; i < axis_dim; ++i) {
        float val = static_cast<float>(input[base_offset + i * inner_dim]);
        sum += expf(val - max_val);
    }
    for (size_t i = 0; i < axis_dim; ++i) {
        float val = static_cast<float>(input[base_offset + i * inner_dim]);
        float result = expf(val - max_val) / sum;
        output[base_offset + i * inner_dim] = static_cast<T>(result);
    }
}

void SoftmaxImpl<Device::CUDA>::execute(Tensor* t, int32_t dev_id){
    const Tensor* x = t->src[0];
    int axis = static_cast<int>(t->op_params[0]); // 支持负轴索引
    int dims  = 0;
    for(int i = 0; i < x->dims.size(); ++i){
        if(x->dims[i] != 0) dims++;
    }
    // axis 需要合法
    if(std::abs(axis) >= dims){
        throw std::runtime_error("cuda::softmax axis is out of range");
    }
    size_t outer_dim = 1;
    size_t inner_dim = 1;
    size_t axis_dim = x->dims[axis];
    for (int i = 0; i < axis; ++i) {
        outer_dim *= x->dims[i];
    }
    if (axis < 0) axis += dims;  
    for (int i = axis + 1; i < dims; ++i) {
        inner_dim *= x->dims[i];
    }
    constexpr int threads = 256;
    size_t numel = t->num_elements();
    int blocks = (numel + threads - 1) / threads;

    dtype::dispatch(t->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      out = static_cast<__half*>(t->data);
            const __half* in1 = static_cast<const __half*>(x->data);
            softmax_kernel<<<blocks, threads>>>(in1, out, outer_dim, axis_dim, inner_dim);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16* out = static_cast<__nv_bfloat16*>(t->data);
            const __nv_bfloat16* in1 = static_cast<const __nv_bfloat16*>(x->data);
            softmax_kernel<<<blocks, threads>>>(in1, out, outer_dim, axis_dim, inner_dim);
        }else{
            throw std::runtime_error("cuda::softmax not implemented");
        }
    });
}

template struct SoftmaxImpl<Device::CUDA>;
}
