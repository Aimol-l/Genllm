#include <stdexcept>
#include <algorithm>
#include "tensor.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cuda/rope.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

// ─── RoPE 应用内核 ─────────────────────────────────────────────────
// 对 Q/K 的前 rope_dim 维应用半旋转编码，剩余维度原样复制
template <typename T>
__global__ void apply_rope_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int64_t B, int64_t n_heads, int64_t seq_len,
    int64_t head_dim, int64_t rope_half, int64_t rope_dim,
    int64_t head_stride, int64_t batch_stride,
    int64_t cos_stride, int64_t start_pos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n_heads * seq_len * head_dim) return;

    int i = idx % head_dim;
    int tmp = idx / head_dim;
    int s = tmp % seq_len;
    tmp /= seq_len;
    int h = tmp % n_heads;
    int b = tmp / n_heads;

    const T* row_in = input + b * batch_stride + h * head_stride + s * head_dim;
    T* row_out = out + b * batch_stride + h * head_stride + s * head_dim;
    int64_t pos = start_pos + s;

    if (i < rope_half) {
        float x0 = float(row_in[i]);
        float x1 = float(row_in[i + rope_half]);
        float c = cos_cache[pos * cos_stride + i];
        float si = sin_cache[pos * cos_stride + i];
        row_out[i] = T(x0 * c - x1 * si);
        row_out[i + rope_half] = T(x0 * si + x1 * c);
    } else if (i >= rope_dim) {
        row_out[i] = row_in[i];
    }
}

// ─── RoPE Cache 内核 ──────────────────────────────────────────────
__global__ void rope_cache_kernel(
    float* dst, float theta, int max_seq, int head_dim, int half_dim, bool is_cos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_seq * head_dim) return;

    int s = idx / head_dim;
    int i = idx % head_dim;
    int half_i = i % half_dim;

    double exponent = (2.0 * half_i) / (double)head_dim;
    double inv_freq = 1.0 / pow((double)theta, exponent);
    double angle = (double)s * inv_freq;
    dst[idx] = is_cos ? (float)cos(angle) : (float)sin(angle);
}

// ─── ApplyRopeImpl ────────────────────────────────────────────────
void ApplyRopeImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    const Tensor* x   = out->src[0]; // [B, n_heads, seq_len, head_dim]
    const Tensor* cos = out->src[1]; // [max_seq_len, rope_dim]
    const Tensor* sin = out->src[2]; // [max_seq_len, rope_dim]

    int64_t head_dim  = static_cast<int64_t>(out->op_params[0]);
    int64_t rope_dim  = static_cast<int64_t>(out->op_params[1]);
    int64_t start_pos = static_cast<int64_t>(out->op_params[2]);

    int64_t rope_half = rope_dim / 2;
    int64_t B = x->dims[0], n_heads = x->dims[1], seq_len = x->dims[2];

    int64_t head_stride  = seq_len * head_dim;
    int64_t batch_stride = n_heads * head_stride;
    int64_t cos_stride   = rope_dim;

    int64_t total = B * n_heads * seq_len * head_dim;
    constexpr int THREADS = 256;
    int blocks = (total + THREADS - 1) / THREADS;

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float>) {
            apply_rope_kernel<<<blocks, THREADS>>>(
                static_cast<float*>(out->data),
                static_cast<const float*>(x->data),
                static_cast<const float*>(cos->data),
                static_cast<const float*>(sin->data),
                B, n_heads, seq_len, head_dim, rope_half, rope_dim,
                head_stride, batch_stride, cos_stride, start_pos
            );
        } else if constexpr (std::is_same_v<T, float16_t>) {
            apply_rope_kernel<<<blocks, THREADS>>>(
                static_cast<__half*>(out->data),
                static_cast<const __half*>(x->data),
                static_cast<const float*>(cos->data),
                static_cast<const float*>(sin->data),
                B, n_heads, seq_len, head_dim, rope_half, rope_dim,
                head_stride, batch_stride, cos_stride, start_pos
            );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            apply_rope_kernel<<<blocks, THREADS>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(x->data),
                static_cast<const float*>(cos->data),
                static_cast<const float*>(sin->data),
                B, n_heads, seq_len, head_dim, rope_half, rope_dim,
                head_stride, batch_stride, cos_stride, start_pos
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

// ─── RopeCacheImpl ────────────────────────────────────────────────
void RopeCacheImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    const float theta = out->op_params[0];
    const int head_dim = static_cast<int>(out->op_params[1]);
    const int max_seq = static_cast<int>(out->op_params[2]);
    const bool is_cos = out->name.find("_cos") != std::string::npos;

    float* dst = static_cast<float*>(out->data);
    const int half_dim = head_dim / 2;
    const int total = max_seq * head_dim;

    constexpr int THREADS = 256;
    const int blocks = (total + THREADS - 1) / THREADS;

    rope_cache_kernel<<<blocks, THREADS>>>(dst, theta, max_seq, head_dim, half_dim, is_cos);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::format("cuda::rope_cache kernel launch failed: {}", cudaGetErrorString(err)));
    }
}

template struct ApplyRopeImpl<Device::CUDA>;
template struct RopeCacheImpl<Device::CUDA>;
}
