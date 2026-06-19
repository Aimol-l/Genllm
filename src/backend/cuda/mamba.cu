#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <cublas_v2.h>
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "backend/cuda/mamba.h"
#include "utils/dtype_traits.hpp"

namespace ops {

// ──────────────────────────────────────────────────────────────────
// Causal Conv1D
//   input       : [seq, D_inner]   row-major (row = token, col = channel)
//   weight      : [D_inner, K]     depthwise filter, order [current,...,oldest]
//   conv_state  : [K-1, D_inner]   row 0 = oldest, row K-2 = newest
//   output      : [seq, D_inner]
// 每个 (t, d) 独立，2D grid 并行
// ──────────────────────────────────────────────────────────────────
template <typename T>
__global__ void causal_conv1d_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const float* __restrict__ weight,
    const T* __restrict__ conv_state,
    int32_t k_size, int32_t seq_len, int32_t D_inner)
{
    int32_t t = blockIdx.x;
    int32_t d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= seq_len || d >= D_inner) return;

    const int32_t pad = k_size - 1;
    const float* w_row = weight + d * k_size;  // [K]
    float acc = 0.f;
    #pragma unroll
    for (int32_t k = 0; k < 32; ++k) {
        if (k >= k_size) break;
        int32_t src_t = t - k;
        float x_val;
        if (src_t >= 0) {
            x_val = float(input[src_t * D_inner + d]);
        } else {
            int32_t state_row = src_t + pad;
            x_val = float(conv_state[state_row * D_inner + d]);
        }
        // weight 布局 [D_inner, K]，w_row[k_size-1-k] = k=0→末尾(current)，对齐 CPU flip
        acc += w_row[k_size - 1 - k] * x_val;
    }
    out[t * D_inner + d] = T(acc);
}

// 更新 conv_state：保留 input 最后 pad 行；若 seq < pad，老状态左移再追加
template <typename T>
__global__ void conv_state_update_kernel(
    T* __restrict__ conv_state,    // [pad, D_inner]
    const T* __restrict__ input,   // [seq, D_inner]
    int32_t pad, int32_t seq_len, int32_t D_inner)
{
    int32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D_inner) return;

    if (seq_len >= pad) {
        // 直接覆盖：conv_state[:, d] = input[seq-pad..seq-1, d]
        const T* src = input + (seq_len - pad) * D_inner + d;
        for (int32_t i = 0; i < pad; ++i) {
            conv_state[i * D_inner + d] = src[i * D_inner];
        }
    } else {
        // 老状态左移 seq 位，再追加新 seq 行
        // 新 conv_state[0..pad-seq-1] = 老 conv_state[seq..pad-1]
        // 新 conv_state[pad-seq..pad-1] = input[0..seq-1]
        for (int32_t i = 0; i < pad - seq_len; ++i) {
            conv_state[i * D_inner + d] = conv_state[(i + seq_len) * D_inner + d];
        }
        for (int32_t i = 0; i < seq_len; ++i) {
            conv_state[(pad - seq_len + i) * D_inner + d] = input[i * D_inner + d];
        }
    }
}

void CausalConv1dImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    Tensor* input      = out->src[0];   // [1, seq, D_inner]
    Tensor* weight     = out->src[1];   // [D_inner, K]
    Tensor* conv_state = out->src[2];   // [1, K-1, D_inner]

    int32_t seq_len = static_cast<int32_t>(input->dims[1]);
    int32_t D_inner = static_cast<int32_t>(input->dims[2]);
    int32_t k_size  = static_cast<int32_t>(out->op_params[0]);

    constexpr int THREADS = 256;
    dim3 grid(seq_len, (D_inner + THREADS - 1) / THREADS);
    dim3 update_grid((D_inner + THREADS - 1) / THREADS);

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            causal_conv1d_kernel<__half><<<grid, THREADS>>>(
                static_cast<__half*>(out->data),
                static_cast<const __half*>(input->data),
                static_cast<const float*>(weight->data),
                static_cast<const __half*>(conv_state->data),
                k_size, seq_len, D_inner);
            conv_state_update_kernel<__half><<<update_grid, THREADS>>>(
                static_cast<__half*>(conv_state->data),
                static_cast<const __half*>(input->data),
                k_size - 1, seq_len, D_inner);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            causal_conv1d_kernel<__nv_bfloat16><<<grid, THREADS>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(input->data),
                static_cast<const float*>(weight->data),
                static_cast<const __nv_bfloat16*>(conv_state->data),
                k_size, seq_len, D_inner);
            conv_state_update_kernel<__nv_bfloat16><<<update_grid, THREADS>>>(
                static_cast<__nv_bfloat16*>(conv_state->data),
                static_cast<const __nv_bfloat16*>(input->data),
                k_size - 1, seq_len, D_inner);
        } else if constexpr (std::is_same_v<T, float>) {
            causal_conv1d_kernel<float><<<grid, THREADS>>>(
                static_cast<float*>(out->data),
                static_cast<const float*>(input->data),
                static_cast<const float*>(weight->data),
                static_cast<const float*>(conv_state->data),
                k_size, seq_len, D_inner);
            conv_state_update_kernel<float><<<update_grid, THREADS>>>(
                static_cast<float*>(conv_state->data),
                static_cast<const float*>(input->data),
                k_size - 1, seq_len, D_inner);
        } else {
            throw std::runtime_error("causal_conv1d<CUDA>: unsupported dtype");
        }
    });
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "causal_conv1d<CUDA> failed: %s\n", cudaGetErrorString(err));
    }
}

// ──────────────────────────────────────────────────────────────────
// SSM Scan (Gated Delta Rule) — 对齐 CPU mamba.cpp gated_delta_rule_ref
//   Q, K       : [seq, num_k_heads * D]   flat row-major
//   V          : [seq, num_v_heads * D]   flat row-major
//   a_log      : [num_v_heads]            ssm_a (per V head)
//   b_data     : [seq, num_v_heads]
//   a_data     : [seq, num_v_heads]
//   dt_bias    : [num_v_heads]
//   state      : [num_v_heads, D, D] F32  (per V head 独立)
//   out        : [seq, num_v_heads * D]
//
// 并行：grid.x = num_v_heads，blockDim.x = D（每线程一列 state）
// ──────────────────────────────────────────────────────────────────
template <typename T>
__global__ void ssm_scan_kernel(
    T* __restrict__ out,
    const T* __restrict__ Q_data,
    const T* __restrict__ K_data,
    const T* __restrict__ V_data,
    const float* __restrict__ a_log,
    const T* __restrict__ b_data,
    const T* __restrict__ a_data,
    const float* __restrict__ dt_bias,
    float* __restrict__ state,
    int32_t seq_len, int32_t num_k_heads, int32_t num_v_heads, int32_t D)
{
    int32_t h = blockIdx.x;               // V head
    int32_t tid = threadIdx.x;            // [0, D)
    if (tid >= D) return;
    int32_t n_rep = num_v_heads / num_k_heads;
    int32_t kh = h / n_rep;              // key head index

    extern __shared__ float smem[];
    float* state_sm  = smem;                  // [D * D]
    float* q_sm      = state_sm + D * D;      // [D]
    float* k_sm      = q_sm + D;              // [D]
    float* v_sm      = k_sm + D;              // [D]

    const int32_t k_row_stride = num_k_heads * D;
    const int32_t v_row_stride = num_v_heads * D;

    // 初始化 state（V head 独立)
    for (int32_t i = tid; i < D * D; i += blockDim.x) {
        state_sm[i] = state[h * D * D + i];
    }
    __syncthreads();

    const float q_scale = 1.0f / sqrtf(static_cast<float>(D));

    for (int32_t t = 0; t < seq_len; ++t) {
        // 1. 载入 q/k/v 到 shared mem（flat layout: [seq, heads*D] row-major）
        const T* q_row = Q_data + (int64_t)t * k_row_stride;
        const T* k_row = K_data + (int64_t)t * k_row_stride;
        const T* v_row = V_data + (int64_t)t * v_row_stride;
        q_sm[tid] = float(q_row[kh * D + tid]);
        k_sm[tid] = float(k_row[kh * D + tid]);
        v_sm[tid] = float(v_row[h  * D + tid]);
        __syncthreads();

        // 2. 求 q_sq, k_sq (block reduce)
        float q_sq = q_sm[tid] * q_sm[tid];
        float k_sq = k_sm[tid] * k_sm[tid];
        for (int32_t i = 0; i < D; ++i) {
            if (i != tid) { q_sq += q_sm[i] * q_sm[i]; k_sq += k_sm[i] * k_sm[i]; }
        }
        // Python: L2norm(Q) then Q *= 1/sqrt(D) → q = Q / (sqrt(D)*sqrt(||Q||²+eps))
        float q_inv = q_scale / sqrtf(q_sq + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_sq + 1e-6f);

        // 归一化
        q_sm[tid] *= q_inv;
        k_sm[tid] *= k_inv;
        __syncthreads();

        // 3. gate: decay = exp(a_log[h] * softplus(a + dt_bias[h]))
        // ssm_a is per-V-head [num_v_heads], not per-K-head
        float b_val = float(b_data[(int64_t)t * num_v_heads + h]);
        float a_val = float(a_data[(int64_t)t * num_v_heads + h]);
        float beta = 1.0f / (1.0f + expf(-b_val));
        float sp = (a_val + dt_bias[h] > 20.0f) ? (a_val + dt_bias[h]) : log1pf(expf(a_val + dt_bias[h]));
        float decay = expf(a_log[h] * sp);

        // 4. state 衰减
        for (int32_t i = 0; i < D; ++i) {
            state_sm[i * D + tid] *= decay;
        }
        __syncthreads();

        // 5. kv_mem[tid] = sum_i state[i, tid] * k[i]
        float kv_mem = 0.f;
        for (int32_t i = 0; i < D; ++i) {
            kv_mem += state_sm[i * D + tid] * k_sm[i];
        }

        // 6. delta = (v - kv_mem) * beta
        float delta = (v_sm[tid] - kv_mem) * beta;

        // 7. state 更新: state[i, tid] += k[i] * delta
        for (int32_t i = 0; i < D; ++i) {
            state_sm[i * D + tid] += k_sm[i] * delta;
        }

        // 8. 输出: out[t, h, tid] = sum_i state[i, tid] * q[i]
        float out_val = 0.f;
        for (int32_t i = 0; i < D; ++i) {
            out_val += state_sm[i * D + tid] * q_sm[i];
        }
        out[(int64_t)t * v_row_stride + h * D + tid] = T(out_val);
        __syncthreads();
    }

    // 写回 state (V head)
    for (int32_t i = tid; i < D * D; i += blockDim.x) {
        state[h * D * D + i] = state_sm[i];
    }
}

void SsmScanImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    Tensor* Q         = out->src[0];   // [1, seq, num_k_heads, D]
    Tensor* K         = out->src[1];   // [1, seq, num_k_heads, D]
    Tensor* V         = out->src[2];   // [1, seq, num_v_heads, D]
    Tensor* ssm_a     = out->src[3];   // A_log [num_k_heads]
    Tensor* b_proj    = out->src[4];   // [1, seq, num_v_heads]
    Tensor* a_proj    = out->src[5];   // [1, seq, num_v_heads]
    Tensor* ssm_dt    = out->src[6];   // dt_bias [num_v_heads]
    Tensor* ssm_state = out->src[7];   // state [1, num_v_heads, D, D] F32

    int32_t num_k_heads = static_cast<int32_t>(out->op_params[0]);
    int32_t D           = static_cast<int32_t>(out->op_params[1]);
    int32_t num_v_heads = static_cast<int32_t>(out->op_params[2]);
    int32_t seq_len     = static_cast<int32_t>(Q->dims[1]);

    int threads = D;
    int blocks  = num_v_heads;
    size_t smem_bytes = (D * D + 3 * D) * sizeof(float);

    // D=128 时需要 opt-in shared mem
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(ssm_scan_kernel<__half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cudaFuncSetAttribute(ssm_scan_kernel<__nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        cudaFuncSetAttribute(ssm_scan_kernel<float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr_set = true;
    }

    dtype::dispatch(out->dtype, [&]<DataType D_w>() {
        using T = dtype::type_t<D_w>;
        if constexpr (std::is_same_v<T, float16_t>) {
            ssm_scan_kernel<__half><<<blocks, threads, smem_bytes>>>(
                static_cast<__half*>(out->data),
                static_cast<const __half*>(Q->data),
                static_cast<const __half*>(K->data),
                static_cast<const __half*>(V->data),
                static_cast<const float*>(ssm_a->data),
                static_cast<const __half*>(b_proj->data),
                static_cast<const __half*>(a_proj->data),
                static_cast<const float*>(ssm_dt->data),
                static_cast<float*>(ssm_state->data),
                seq_len, num_k_heads, num_v_heads, D);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            ssm_scan_kernel<__nv_bfloat16><<<blocks, threads, smem_bytes>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(Q->data),
                static_cast<const __nv_bfloat16*>(K->data),
                static_cast<const __nv_bfloat16*>(V->data),
                static_cast<const float*>(ssm_a->data),
                static_cast<const __nv_bfloat16*>(b_proj->data),
                static_cast<const __nv_bfloat16*>(a_proj->data),
                static_cast<const float*>(ssm_dt->data),
                static_cast<float*>(ssm_state->data),
                seq_len, num_k_heads, num_v_heads, D);
        } else if constexpr (std::is_same_v<T, float>) {
            ssm_scan_kernel<float><<<blocks, threads, smem_bytes>>>(
                static_cast<float*>(out->data),
                static_cast<const float*>(Q->data),
                static_cast<const float*>(K->data),
                static_cast<const float*>(V->data),
                static_cast<const float*>(ssm_a->data),
                static_cast<const float*>(b_proj->data),
                static_cast<const float*>(a_proj->data),
                static_cast<const float*>(ssm_dt->data),
                static_cast<float*>(ssm_state->data),
                seq_len, num_k_heads, num_v_heads, D);
        } else {
            throw std::runtime_error("ssm_scan<CUDA>: unsupported dtype");
        }
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ssm_scan<CUDA> failed: %s\n", cudaGetErrorString(err));
    }
}

template struct CausalConv1dImpl<Device::CUDA>;
template struct SsmScanImpl<Device::CUDA>;

} // namespace ops
