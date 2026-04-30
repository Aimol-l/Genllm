#include <stdexcept>
#include "backend/cuda/normalization.h"
#include "utils/dtype_traits.hpp"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

// rms = x[1,seq,2560] -平方-> [1,seq,2560] -mean-> [1,seq,1] -sqrt-> [1,seq,1] 
// out = (x/rms) * w[2560]
template<typename T>
__global__ void rmsnorm_warp_kernel(
    T* __restrict__ out,
    const T* __restrict__ x ,         // [1,seq,hidden]
    const float* __restrict__ weight, // [hidden]
    int rows, int hidden_size, float eps
){
    // 每个 warp 处理一行
    int warp_id = threadIdx.x / 32; // 因为一个warp是32个线程，所以每个warp处理一行
    int lane_id = threadIdx.x % 32; // 算出当前线程在warp中的编号
    int row = blockIdx.x * (blockDim.x / 32) + warp_id; // 算出当前线程处理的行号，共rows行
    if (row >= rows) return; // 超出行数，直接返回

    T* out_row = out + row * hidden_size;   // 输出地址
    const T* x_row = x + row * hidden_size; // 输入地址

    // ===== Step 1: 计算 sum(x^2) =====
    float sum = 0.f;
    // 32 个线程交错读取整个 hidden_size 维度
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
        float o = v * inv_rms * weight[i];
        out_row[i] = T(o);
    }
}
void RmsNormImpl<Device::CUDA>::execute(Tensor* out){
    const Tensor* x = out->src[0];  // [B, seq, hidden] 或 [B, n_heads, seq, head_dim]
    const Tensor* w = out->src[1];  // [hidden] or [head_dim]
    float eps = out->op_params[0];
    int64_t hidden_size = static_cast<int64_t>(w->num_elements());
    int64_t total_elems = static_cast<int64_t>(x->num_elements());
    int rows = static_cast<int>(total_elems / hidden_size);
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
            rmsnorm_warp_kernel<<<blocks, THREADS>>>(ou,in,weight,rows,hidden_size,eps);
        }else if constexpr(std::is_same_v<T,bfloat16_t>){
            __nv_bfloat16*      ou = static_cast<__nv_bfloat16*>(out->data);
            const __nv_bfloat16* in = static_cast<const __nv_bfloat16*>(x->data);
            const float* weight = static_cast<const float*>(w->data);
            rmsnorm_warp_kernel<<<blocks, THREADS>>>(ou,in,weight,rows,hidden_size,eps);
        }else if constexpr(std::is_same_v<T,float>){
            float*      ou = static_cast<float*>(out->data);
            const float* in = static_cast<const float*>(x->data);
            const float* weight = static_cast<const float*>(w->data);
            rmsnorm_warp_kernel<<<blocks, THREADS>>>(ou,in,weight,rows,hidden_size,eps);
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
