#include <cstddef>
#include <stdexcept>
#include "tensor.hpp"
#include "backend/cuda/attention.h"
#include "utils.hpp"
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

void SoftmaxImpl<Device::CUDA>::execute(Tensor* t){
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
        outer_dim *= x->dims[axis];
    }
    if (axis < 0) axis += dims;  
    for (int i = axis + 1; i < dims; ++i) {
        inner_dim *= x->dims[axis];
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


void AttentionImpl<Device::CUDA>::execute(Tensor*)       { throw std::runtime_error("cuda::attention not implemented"); }


template <typename T, int HEAD_DIM=128>
__global__ void flash_attention_warp_kernel(
    T* __restrict__ out,
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    int64_t B, int64_t n_heads, int64_t Sq,
    int64_t n_kv_heads, int64_t Skv,
    float scale_val, int causal, int num_kv_groups
) {
    const int bh = blockIdx.x;
    if (bh >= B * n_heads) return;
    const int head_idx  = bh % n_heads;
    const int batch_idx = bh / n_heads;
    const int kv_head_idx = head_idx / num_kv_groups;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int q_pos = warp_id + blockIdx.y * static_cast<int>(blockDim.x / 32);
    if (q_pos >= Sq) return;

    const int64_t q_base  = (batch_idx * n_heads + head_idx) * Sq * HEAD_DIM;
    const int64_t kv_base = (batch_idx * n_kv_heads + kv_head_idx) * Skv * HEAD_DIM;
    const int64_t out_base = q_base;

    constexpr int KV_TILE = 32;
    extern __shared__ char smem_[];
    T* shared_K = reinterpret_cast<T*>(smem_);
    T* shared_V = reinterpret_cast<T*>(smem_ + KV_TILE * HEAD_DIM * sizeof(T));

    constexpr int VEC = HEAD_DIM / 32;
    float q_reg[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int d_idx = lane_id + i * 32;
        q_reg[i] = float(Q[q_base + q_pos * HEAD_DIM + d_idx]);
    }
    float m_i = -INFINITY;
    float l_i = 0.f;
    float o_reg[VEC] = {0};

    for (int kv_tile = 0; kv_tile < Skv; kv_tile += KV_TILE) {
        int tile_size = KV_TILE < static_cast<int>(Skv - kv_tile) ? KV_TILE : static_cast<int>(Skv - kv_tile);

        for (int i = threadIdx.x; i < tile_size * HEAD_DIM; i += blockDim.x) {
            int k = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            shared_K[i] = K[kv_base + (kv_tile + k) * HEAD_DIM + d];
            shared_V[i] = V[kv_base + (kv_tile + k) * HEAD_DIM + d];
        }
        __syncthreads();

        for (int k = 0; k < tile_size; ++k) {
            int kv_pos = kv_tile + k;
            if (causal && kv_pos > q_pos) continue;

            float score_partial = 0.f;
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int d_idx = lane_id + i * 32;
                score_partial += q_reg[i] * float(shared_K[k * HEAD_DIM + d_idx]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                score_partial += __shfl_down_sync(0xffffffff, score_partial, offset);
            }
            float score = __shfl_sync(0xffffffff, score_partial, 0);
            score *= scale_val;

            float m_prev = m_i;
            m_i = fmaxf(m_i, score);
            float exp_score = __expf(score - m_i);
            float exp_prev  = __expf(m_prev - m_i);
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int d_idx = lane_id + i * 32;
                float v = float(shared_V[k * HEAD_DIM + d_idx]);
                o_reg[i] = o_reg[i] * exp_prev + exp_score * v;
            }
            l_i = l_i * exp_prev + exp_score;
        }
        __syncthreads();
    }

    float inv_l = 1.f / l_i;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        int d_idx = lane_id + i * 32;
        out[out_base + q_pos * HEAD_DIM + d_idx] = T(o_reg[i] * inv_l);
    }
}
void FlashAttentionImpl<Device::CUDA>::execute(Tensor* out) {
    const Tensor* Q = out->src[0];
    const Tensor* K = out->src[1];
    const Tensor* V = out->src[2];

    // ===== 参数 =====
    int64_t head_dim = static_cast<int64_t>(out->op_params[0]);
    float scale_val  = out->op_params[1];
    int32_t causal   = static_cast<int32_t>(out->op_params[2]);
    int32_t num_kv_groups = static_cast<int32_t>(out->op_params[3]);

    int64_t B = Q->dims[0];
    int64_t n_heads = Q->dims[1];
    int64_t Sq = Q->dims[2];

    int64_t n_kv_heads = K->dims[1];
    int64_t Skv = K->dims[2];

    // ===== 强约束（当前 kernel 版本）=====
    constexpr int HEAD_DIM = 128;
    if (head_dim != HEAD_DIM) {
        throw std::runtime_error("flash_attention_warp_kernel requires head_dim == 128");
    }
    if (Sq <= 0 || Skv <= 0) {
        throw std::runtime_error("invalid sequence length");
    }
    // ===== kernel 配置 =====
    constexpr int KV_TILE = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int blocks_x = static_cast<int>(B * n_heads);
    const int blocks_y = static_cast<int>((Sq + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const dim3 grid_dim(blocks_x, blocks_y);
    const size_t shared_mem = 2 * KV_TILE * HEAD_DIM * sizeof(__nv_bfloat16);
    // ===== launch =====
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            auto* out_ptr = static_cast<__half*>(out->data);
            auto* Q_ptr   = static_cast<const __half*>(Q->data);
            auto* K_ptr   = static_cast<const __half*>(K->data);
            auto* V_ptr   = static_cast<const __half*>(V->data);
            flash_attention_warp_kernel<__half, HEAD_DIM>
                <<<grid_dim, THREADS, shared_mem>>>(
                    out_ptr,
                    Q_ptr,
                    K_ptr,
                    V_ptr,
                    B, n_heads, Sq,
                    n_kv_heads, Skv,
                    scale_val,
                    causal,
                    num_kv_groups
                );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            auto* out_ptr = static_cast<__nv_bfloat16*>(out->data);
            auto* Q_ptr   = static_cast<const __nv_bfloat16*>(Q->data);
            auto* K_ptr   = static_cast<const __nv_bfloat16*>(K->data);
            auto* V_ptr   = static_cast<const __nv_bfloat16*>(V->data);
            flash_attention_warp_kernel<__nv_bfloat16, HEAD_DIM>
                <<<grid_dim, THREADS, shared_mem>>>(
                    out_ptr,
                    Q_ptr,
                    K_ptr,
                    V_ptr,
                    B, n_heads, Sq,
                    n_kv_heads, Skv,
                    scale_val,
                    causal,
                    num_kv_groups
                );
        } else {
            throw std::runtime_error("FlashAttention only supports fp16/bf16");
        }
    });
    // ===== error check =====
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "flash_attention_warp_kernel launch failed: %s\n",
            cudaGetErrorString(err));
    }
}

void SdpaImpl<Device::CUDA>::execute(Tensor* out){
    FlashAttentionImpl<Device::CUDA>::execute(out);
}
void DiagMaskInfImpl<Device::CUDA>::execute(Tensor*)     { throw std::runtime_error("cuda::diag_mask_inf not implemented"); }


template struct SoftmaxImpl<Device::CUDA>;
template struct DiagMaskInfImpl<Device::CUDA>;
template struct SdpaImpl<Device::CUDA>;
template struct AttentionImpl<Device::CUDA>;
template struct FlashAttentionImpl<Device::CUDA>;
}
