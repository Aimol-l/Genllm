#include <cmath>
#include <stdexcept>
#include "tensor.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cuda/rope.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

template <typename T>
__global__ void apply_rope_kernel(
    T* __restrict__ out,
    const T* __restrict__ inp,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int64_t head_dim,
    int64_t half_dim,
    int64_t seq_len,
    int64_t start_pos,
    int64_t n_heads,
    int64_t B,
    size_t batch_stride,
    size_t head_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t n_pairs = B * n_heads * seq_len * half_dim;

    for (int64_t p = idx; p < n_pairs; p += stride) {
        int64_t i = p % half_dim;
        int64_t s = (p / half_dim) % seq_len;
        int64_t h = (p / (half_dim * seq_len)) % n_heads;
        int64_t b = p / (half_dim * seq_len * n_heads);

        int64_t pos = start_pos + s;
        int64_t offset = b * static_cast<int64_t>(batch_stride)
                       + h * static_cast<int64_t>(head_stride)
                       + s * head_dim;

        float x0 = static_cast<float>(inp[offset + i]);
        float x1 = static_cast<float>(inp[offset + i + half_dim]);
        float c = cos_cache[pos * head_dim + i];
        float si = sin_cache[pos * head_dim + i];

        out[offset + i] = static_cast<T>(x0 * c - x1 * si);
        out[offset + i + half_dim] = static_cast<T>(x0 * si + x1 * c);
    }
}

void ApplyRopeImpl<Device::CUDA>::execute(Tensor* out) {
    const Tensor* x   = out->src[0];
    const Tensor* cos = out->src[1];
    const Tensor* sin = out->src[2];

    int64_t head_dim  = static_cast<int64_t>(out->op_params[0]);
    int64_t half_dim  = head_dim / 2;
    int64_t start_pos = static_cast<int64_t>(out->op_params[2]);

    int64_t B = x->dims[0], n_heads = x->dims[1], seq_len = x->dims[2];
    size_t head_stride  = static_cast<size_t>(seq_len * head_dim);
    size_t batch_stride = static_cast<size_t>(n_heads * seq_len * head_dim);

    constexpr int threads = 256;
    int64_t n_pairs = B * n_heads * seq_len * half_dim;
    int blocks = static_cast<int>((n_pairs + threads - 1) / threads);

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float>) {
            apply_rope_kernel<<<blocks, threads>>>(
                static_cast<float*>(out->data),
                static_cast<const float*>(x->data),
                static_cast<const float*>(cos->data),
                static_cast<const float*>(sin->data),
                head_dim, half_dim, seq_len, start_pos, n_heads, B,
                batch_stride, head_stride
            );
        } else if constexpr (std::is_same_v<T, float16_t>) {
            apply_rope_kernel<<<blocks, threads>>>(
                static_cast<__half*>(out->data),
                static_cast<const __half*>(x->data),
                static_cast<const float*>(cos->data),
                static_cast<const float*>(sin->data),
                head_dim, half_dim, seq_len, start_pos, n_heads, B,
                batch_stride, head_stride
            );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            apply_rope_kernel<<<blocks, threads>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(x->data),
                static_cast<const float*>(cos->data),
                static_cast<const float*>(sin->data),
                head_dim, half_dim, seq_len, start_pos, n_heads, B,
                batch_stride, head_stride
            );
        } else {
            throw std::runtime_error("cuda::apply_rope: unsupported dtype");
        }
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::format("cuda::apply_rope kernel launch failed: {}", cudaGetErrorString(err)));
    }
}

template struct ApplyRopeImpl<Device::CUDA>;
}
