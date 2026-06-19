#include <cstring>
#include <stdexcept>
#include <vector>
#include "core/tensor.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cuda/shape.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

template <typename T, const int MAX_DIM = TENSOR_MAX_DIMS>
__global__ void permute_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    const uint64_t* in_strides,
    const uint64_t* out_strides,
    const int64_t* axes,
    int ndim,
    size_t numel
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numel) return;
    // 计算多维坐标 old_idx
    int coord[MAX_DIM]; // 假设 <= 8 维
    int tmp = tid;
    for (int i = 0; i < ndim; ++i) {
        coord[i] = tmp / in_strides[i];
        tmp %= in_strides[i];
    }
    // 生成新坐标 new_coord
    int new_coord[MAX_DIM];
    for (int i = 0; i < ndim; ++i) {
        new_coord[i] = coord[axes[i]];
    }
    // 计算输出位置
    int out_index = 0;
    for (int i = 0; i < ndim; ++i) {
        out_index += new_coord[i] * out_strides[i];
    }
    dst[out_index] = src[tid];
}

void PermuteImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    const Tensor* x = out->src[0]; // [batch,seq,head,dim]

    // 有效维度数
    int ndim = 0;
    for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) {
        ndim = i + 1;
    }

    // 如果 ndim == 1, 就是恒等映射，无需计算
    if (ndim <= 1) return;

    int64_t* d_axes;
    uint64_t* d_in_strides, * d_out_strides;
    cudaMallocManaged(&d_axes, sizeof(int64_t) * ndim);
    cudaMallocManaged(&d_in_strides, sizeof(uint64_t) * ndim);
    cudaMallocManaged(&d_out_strides, sizeof(uint64_t) * ndim);

    for(int i = 0; i < ndim; ++i){
        d_in_strides[i] = x->strides[i];
        d_out_strides[i] = out->strides[i];
        d_axes[i] = static_cast<int64_t>(out->op_params[i]);
    }

    constexpr int threads = 256;
    size_t total = out->num_elements();
    int blocks = static_cast<int>((total + threads - 1) / threads);

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float>) {
            permute_kernel<<<blocks, threads>>>(
                static_cast<float*>(out->data),
                static_cast<const float*>(x->data),
                d_in_strides, d_out_strides, d_axes, ndim, total
            );
        } else if constexpr (std::is_same_v<T, float16_t>) {
            permute_kernel<<<blocks, threads>>>(
                static_cast<__half*>(out->data),
                static_cast<const __half*>(x->data),
                d_in_strides, d_out_strides, d_axes, ndim, total
            );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            permute_kernel<<<blocks, threads>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(x->data),
                d_in_strides, d_out_strides, d_axes, ndim, total
            );
        } else {
            cudaFree(d_in_strides);
            cudaFree(d_out_strides);
            cudaFree(d_axes);
            throw std::runtime_error("cuda::permute: unsupported dtype");
        }
    });
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
    cudaFree(d_axes);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::format("cuda::permute kernel launch failed: {}", cudaGetErrorString(err)));
    }
}

void ReshapeImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    const Tensor* x = out->src[0];
    out->data   = x->data;
    out->offset = x->offset;
    size_t stride = 1;
    for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
        if (out->dims[i] == 0) {
            out->strides[i] = 0;
        } else {
            out->strides[i] = stride;
            stride *= static_cast<size_t>(out->dims[i]);
        }
    }
}
void ConcatImpl<Device::CUDA>::execute(Tensor*, int32_t)  { throw std::runtime_error("cuda::concat not implemented"); }
void RepeatImpl<Device::CUDA>::execute(Tensor*, int32_t)  { throw std::runtime_error("cuda::repeat not implemented"); }

void SplitImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    const Tensor* x = out->src[0];
    int32_t dim   = static_cast<int32_t>(out->op_params[0]);
    int64_t start = static_cast<int64_t>(out->op_params[1]);
    int64_t size  = static_cast<int64_t>(out->op_params[2]);
    size_t elem_sz = data_type_size(out->dtype);

    int64_t outer = 1;
    for (int d = 0; d < dim; ++d)
        if (x->dims[d] > 0) outer *= x->dims[d];

    size_t src_block_bytes = 1;
    for (int d = dim; d < TENSOR_MAX_DIMS && x->dims[d] > 0; ++d)
        src_block_bytes *= x->dims[d];
    src_block_bytes *= elem_sz;

    size_t dst_block_bytes = size * elem_sz;
    size_t copy_offset     = start * elem_sz;

    auto* dst = static_cast<uint8_t*>(out->data);
    auto* src = static_cast<const uint8_t*>(x->data);

    for (int64_t i = 0; i < outer; ++i) {
        cudaMemcpy(
            dst + i * dst_block_bytes,
            src + i * src_block_bytes + copy_offset,
            dst_block_bytes,
            cudaMemcpyDeviceToDevice
        );
    }
}

template struct ReshapeImpl<Device::CUDA>;
template struct PermuteImpl<Device::CUDA>;
template struct ConcatImpl<Device::CUDA>;
template struct RepeatImpl<Device::CUDA>;
template struct SplitImpl<Device::CUDA>;
}
