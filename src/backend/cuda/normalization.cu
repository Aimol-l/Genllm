#include <stdexcept>
#include "backend/cuda/normalization.h"
#include "utils/dtype_traits.hpp"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

template<typename T>
__global__ void rmsnorm_warp_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    const float* __restrict__ w, // fp32
    int rows, int hidden_size,
    float eps
){
    // 每个 warp 处理一行
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int row = blockIdx.x * (blockDim.x / 32) + warp_id;
    if (row >= rows) return;

    const T* x_row = x + row * hidden_size;
    T* out_row = out + row * hidden_size;

    // ===== Step 1: 计算 sum(x^2) =====
    float sum = 0.f;

    for (int i = lane_id; i < hidden_size; i += 32) {
        float v = float(x_row[i]);
        sum += v * v;
    }

    // warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    float mean_sq = __shfl_sync(0xffffffff, sum, 0) / hidden_size;

    float inv_rms = rsqrtf(mean_sq + eps);

    // ===== Step 2: normalize + scale =====
    for (int i = lane_id; i < hidden_size; i += 32) {
        float v = float(x_row[i]);
        float o = v * inv_rms * w[i];
        out_row[i] = T(o);
    }
}
void RmsNormImpl<Device::CUDA>::execute(Tensor* out){
    const Tensor* x = out->src[0];  // [B, S, H]
    const Tensor* w = out->src[1];  // [H] fp32
    float eps = out->op_params[0];
    int64_t B = x->dims[0];
    int64_t S = x->dims[1];
    int64_t H = x->dims[2];
    int rows = B * S;
    // ===== kernel 配置 =====
    constexpr int WARPS_PER_BLOCK = 4;   // 可调
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int blocks = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T,float16_t>) {
            __half*      ou = static_cast<__half*>(out->data);
            const __half* in = static_cast<const __half*>(x->data);
            const float* weight = static_cast<const float*>(w->data);
            rmsnorm_warp_kernel<<<blocks, THREADS>>>(ou,in,weight,rows,H,eps);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16*      ou = static_cast<__nv_bfloat16*>(out->data);
            const __nv_bfloat16* in = static_cast<const __nv_bfloat16*>(x->data);
            const float* weight = static_cast<const float*>(w->data);
            rmsnorm_warp_kernel<<<blocks, THREADS>>>(ou,in,weight,rows,H,eps);
        }else if constexpr(std::is_same_v<T,float>){
            float*      ou = static_cast<float*>(out->data);
            const float* in = static_cast<const float*>(x->data);
            const float* weight = static_cast<const float*>(w->data);
            rmsnorm_warp_kernel<<<blocks, THREADS>>>(ou,in,weight,rows,H,eps);
        }else{
            throw std::runtime_error("RMSNorm only supports fp32/fp16/bf16");
        }
    });
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "rmsnorm kernel failed: %s\n", cudaGetErrorString(err));
    }
}
void LayerNormImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::layer_norm not implemented"); }

template struct RmsNormImpl<Device::CUDA>;
template struct LayerNormImpl<Device::CUDA>;
}
