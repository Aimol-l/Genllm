#include <cstring>
#include <stdexcept>
#include "tensor.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cuda/shape.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

template <typename T>
__global__ void permute_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const int64_t* __restrict__ out_dims,
    const uint64_t* __restrict__ src_strides,
    const int* __restrict__ perm,
    int ndim,
    size_t total_elements
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        size_t remaining = idx;
        size_t src_byte_offset = 0;
        #pragma unroll
        for (int d = ndim - 1; d >= 0; --d) {
            size_t dim_sz = static_cast<size_t>(out_dims[d]);
            size_t coord = remaining % dim_sz;
            remaining /= dim_sz;
            src_byte_offset += coord * src_strides[perm[d]];
        }
        dst[idx] = src[src_byte_offset / sizeof(T)];
    }
}

void PermuteImpl<Device::CUDA>::execute(Tensor* out) {
    const Tensor* x = out->src[0];

    // 有效维度数
    int ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) {
        ndim = i + 1;
    }

    size_t elem_sz = data_type_size(out->dtype);

    // 如果 ndim == 1, 就是恒等映射，无需计算
    if (ndim <= 1) return;

    constexpr int threads = 256;
    size_t total = out->num_elements();
    int blocks = static_cast<int>((total + threads - 1) / threads);

    // 拷贝元数据到设备端：dims, strides, perm
    int64_t h_out_dims[TENSOR_MAX_DIMS];
    uint64_t h_src_strides[TENSOR_MAX_DIMS];
    int h_perm[TENSOR_MAX_DIMS];

    for (int i = 0; i < ndim; ++i) {
        h_out_dims[i] = out->dims[i];
        h_src_strides[i] = x->strides[static_cast<int>(out->op_params[i])];
        h_perm[i] = static_cast<int>(out->op_params[i]);
    }

    int64_t* d_out_dims = nullptr;
    uint64_t* d_src_strides = nullptr;
    int* d_perm = nullptr;

    cudaMalloc(&d_out_dims, ndim * sizeof(int64_t));
    cudaMalloc(&d_src_strides, ndim * sizeof(uint64_t));
    cudaMalloc(&d_perm, ndim * sizeof(int));

    cudaMemcpy(d_out_dims, h_out_dims, ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src_strides, h_src_strides, ndim * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_perm, h_perm, ndim * sizeof(int), cudaMemcpyHostToDevice);

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float>) {
            permute_kernel<<<blocks, threads>>>(
                static_cast<float*>(out->data),
                static_cast<const float*>(x->data),
                d_out_dims, d_src_strides, d_perm, ndim, total
            );
        } else if constexpr (std::is_same_v<T, float16_t>) {
            permute_kernel<<<blocks, threads>>>(
                static_cast<__half*>(out->data),
                static_cast<const __half*>(x->data),
                d_out_dims, d_src_strides, d_perm, ndim, total
            );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            permute_kernel<<<blocks, threads>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(x->data),
                d_out_dims, d_src_strides, d_perm, ndim, total
            );
        } else {
            cudaFree(d_out_dims);
            cudaFree(d_src_strides);
            cudaFree(d_perm);
            throw std::runtime_error("cuda::permute: unsupported dtype");
        }
    });

    cudaFree(d_out_dims);
    cudaFree(d_src_strides);
    cudaFree(d_perm);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::format("cuda::permute kernel launch failed: {}", cudaGetErrorString(err)));
    }
}

void ReshapeImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::reshape not implemented"); }
void ConcatImpl<Device::CUDA>::execute(Tensor*)  { throw std::runtime_error("cuda::concat not implemented"); }
void RepeatImpl<Device::CUDA>::execute(Tensor*)  { throw std::runtime_error("cuda::repeat not implemented"); }

template struct ReshapeImpl<Device::CUDA>;
template struct PermuteImpl<Device::CUDA>;
template struct ConcatImpl<Device::CUDA>;
template struct RepeatImpl<Device::CUDA>;
}
