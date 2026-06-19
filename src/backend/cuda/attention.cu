#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "backend/cuda/attention.h"

#include "float16.hpp"
#include "utils/utils.hpp"
#include "utils/dtype_traits.hpp"
#include "core/manager.h"
#include "core/page_attention.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

#define Br 16
#define Bc 64
#define WARP_SIZE 32

// gridDim = (ceil(seq_len / Br),num_q_heads), blockDim = (512)
__global__ void flash_attention(
    __half* __restrict__ output,
    const __half* __restrict__ Q,
    const __half* __restrict__ K,   // 不要转置它！！！！
    const __half* __restrict__ V,
    int32_t seq_len, int32_t head_dim, int32_t num_q_heads,
    int32_t num_kv_heads, int32_t num_groups, float scale
) {
    // 1. 确定头索引
    int q_head_idx = blockIdx.y;                // 明确当前是哪个查询头，一共是有num_q_heads个
    int kv_head_idx = q_head_idx / num_groups;  // 明确当前是哪个kv头，这样处理是为了适配GQA.多个kv头共用一个q头的情况。
    int q_block_start = blockIdx.x * Br;        // 这个block处理的Q起始行，一共是有 seq/Br 个分块的

    // 2. 指针偏移
    const __half* Q_ptr = Q + q_head_idx * seq_len * head_dim;
    const __half* K_ptr = K + kv_head_idx * seq_len * head_dim;
    const __half* V_ptr = V + kv_head_idx * seq_len * head_dim;
    __half*     out_ptr = output + q_head_idx * seq_len * head_dim;


    // 3. 动态共享内存布局
    extern __shared__ float smem[];                            // 总大小需在启动时指定
    __half* Q_block = (__half*)smem;                           // [Br, head_dim]
    __half* K_block = Q_block + Br * head_dim;                 // [Bc, head_dim]
    __half* V_block = K_block + Bc * head_dim;                 // [Bc, head_dim]
    float* N_acc = (float*)(V_block + Bc * head_dim);          // [Br, head_dim]
    float* D_acc = N_acc + Br * head_dim;                      // [Br]
    float* m = D_acc + Br;                                     // [Br]

    // 4. 线程与 warp 索引
    int tid = threadIdx.x;                  // 确定当前线程: 0...511
    int warp_id = tid / WARP_SIZE;          // 0..511 / 32 = 0..15 ,确定该线程属于哪个 warp（0 到 15）
    int lane_id = tid % WARP_SIZE;          // 0..31,确定该线程在 warp 内的位置（0 到 31）

    int q_row = q_block_start + warp_id;    // 确定当前是处理哪行，起始+局部偏移（用warpid）

    // 5. 从全局显存里面加载切片到加SM的共享内存里面(Q_block,由所有线程协作),每个SM都要加载
    for (int i = tid; i < Br * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        int global_row = q_block_start + row;
        if (global_row < seq_len) {
            Q_block[i] = Q_ptr[global_row * head_dim + col];
        } else {
            Q_block[i] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // 6. 初始化统计量（仅本行的warp负责初始化自己的状态）
    if (warp_id < Br && q_row < seq_len) {
        m[warp_id] = -INFINITY;
        D_acc[warp_id] = 0.0f;
        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            N_acc[warp_id * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    // 7. 外层循环遍历所有 KV 块
    for (int kv_start = 0; kv_start < seq_len; kv_start += Bc) {
        // 加载 K_block 和 V_block 到共享内存（所有线程协作）
        for (int i = tid; i < Bc * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int global_row = kv_start + row;
            if (global_row < seq_len) {
                K_block[i] = K_ptr[global_row * head_dim + col];
                V_block[i] = V_ptr[global_row * head_dim + col];
            } else {
                K_block[i] = __float2half(0.0f);
                V_block[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // 16个warp分别处理16行
        if (warp_id < Br && q_row < seq_len) {
            // 计算当前 warp 负责的 query 行与 K_block 的点积，得到 Bc 个分数; [1,head_dim] @ [Bc,head_dim].T = [1,Bc]
            float scores[Bc];   // 存到寄存器，可能较大，也可以分块计算
            // score[j] 是 warp里面所有线程合作计算得到的结果。128/32 = 4,所以是一个线程算4个a*b=c,最后sum(C)
            for (int j = 0; j < Bc; ++j) {
                float dot = 0.0f;
                for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
                    float q_val = __half2float(Q_block[warp_id * head_dim + d]);
                    float k_val = __half2float(K_block[j * head_dim + d]);
                    dot += q_val * k_val;
                }
                // warp 内归约求和
                for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                scores[j] = dot * scale;
            } 

            // 找到当前块内的最大值 row_max
            float row_max = -INFINITY;
            for (int j = 0; j < Bc; ++j) 
                row_max = fmaxf(row_max, scores[j]);

            float old_m = m[warp_id];
            float new_m = fmaxf(old_m, row_max);

            float old_scale = expf(old_m - new_m);   // 若old_m为 -inf 则 exp(-inf)=0
            float old_D = 0.0f;
            for (int j = 0; j < Bc; ++j) {
                old_D += expf(scores[j] - new_m);
            }
            // 更新分母
            float new_D = D_acc[warp_id] * old_scale + old_D;
            // 更新分子 N_acc（按维度）
            for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
                float acc = N_acc[warp_id * head_dim + d] * old_scale;
                for (int j = 0; j < Bc; ++j) {
                    float p = expf(scores[j] - new_m);
                    float v_val = __half2float(V_block[j * head_dim + d]);
                    acc += p * v_val;
                }
                N_acc[warp_id * head_dim + d] = acc;
            }
            // 更新 m 和 D_acc
            m[warp_id] = new_m;
            D_acc[warp_id] = new_D;
        }
        __syncthreads();
    } // 结束 KV 块循环

    // 8. 最终归一化并写回输出
    if (warp_id < Br && q_row < seq_len) {
        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            float val = N_acc[warp_id * head_dim + d] / D_acc[warp_id];
            out_ptr[q_row * head_dim + d] = __float2half(val);
        }
    }
}
template <typename T, int HEAD_DIM=128>
__global__ void flash_attention_warp_kernel(
    T* __restrict__ out,
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    int64_t B, int64_t num_q_heads, int64_t seq_len,int64_t num_kv_heads,
    float scale_val, int causal, int num_kv_groups
) {
    const int bh = blockIdx.x;
    if (bh >= B * num_q_heads) return;
    const int head_idx  = bh % num_q_heads;
    const int batch_idx = bh / num_q_heads;
    const int kv_head_idx = head_idx / num_kv_groups;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int q_pos = warp_id + blockIdx.y * static_cast<int>(blockDim.x / 32);
    if (q_pos >= seq_len) return;

    const int64_t q_base  = (batch_idx * num_q_heads + head_idx) * seq_len * HEAD_DIM;
    const int64_t kv_base = (batch_idx * num_kv_heads + kv_head_idx) * seq_len * HEAD_DIM;
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

    for (int kv_tile = 0; kv_tile < seq_len; kv_tile += KV_TILE) {
        int tile_size = KV_TILE < static_cast<int>(seq_len - kv_tile) ? KV_TILE : static_cast<int>(seq_len - kv_tile);

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

void FlashAttentionImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id) {
    cudaSetDevice(dev_id);
    const Tensor* Q = out->src[0];
    const Tensor* K = out->src[1];
    const Tensor* V = out->src[2];

    // ===== 参数 =====
    int64_t head_dim = static_cast<int64_t>(out->op_params[0]);
    float scale_val  = out->op_params[1];
    int32_t causal   = static_cast<int32_t>(out->op_params[2]);
    int32_t num_kv_groups = static_cast<int32_t>(out->op_params[3]);

    int64_t B = Q->dims[0];
    int64_t num_q_heads = Q->dims[1];
    int64_t seq_len = Q->dims[2];

    int64_t num_kv_heads = K->dims[1];

    // ===== 强约束（当前 kernel 版本）=====
    if (head_dim != 128 && head_dim != 256) {
        throw std::runtime_error("flash_attention_warp_kernel requires head_dim in {128, 256}");
    }
    if (seq_len <= 0 || seq_len <= 0) {
        throw std::runtime_error("invalid sequence length");
    }
    // ===== kernel 配置 =====
    constexpr int KV_TILE = 32;
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    const int blocks_x = static_cast<int>(B * num_q_heads);
    const int blocks_y = static_cast<int>((seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const dim3 grid_dim(blocks_x, blocks_y);
    const size_t shared_mem_128 = 2 * KV_TILE * 128 * sizeof(__nv_bfloat16);
    const size_t shared_mem_256 = 2 * KV_TILE * 256 * sizeof(__nv_bfloat16);

    // ===== launch with runtime head_dim dispatch =====
    auto launch_kernel = [&]<int HD>() {
        const size_t smem = HD == 256 ? shared_mem_256 : shared_mem_128;
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            if constexpr (std::is_same_v<T, float16_t>) {
                flash_attention_warp_kernel<__half, HD>
                    <<<grid_dim, THREADS, smem>>>(
                        static_cast<__half*>(out->data),
                        static_cast<const __half*>(Q->data),
                        static_cast<const __half*>(K->data),
                        static_cast<const __half*>(V->data),
                        B, num_q_heads, seq_len,
                        num_kv_heads, scale_val, causal, num_kv_groups);
            } else if constexpr (std::is_same_v<T, bfloat16_t>) {
                flash_attention_warp_kernel<__nv_bfloat16, HD>
                    <<<grid_dim, THREADS, smem>>>(
                        static_cast<__nv_bfloat16*>(out->data),
                        static_cast<const __nv_bfloat16*>(Q->data),
                        static_cast<const __nv_bfloat16*>(K->data),
                        static_cast<const __nv_bfloat16*>(V->data),
                        B, num_q_heads, seq_len,
                        num_kv_heads, scale_val, causal, num_kv_groups);
            } else {
                throw std::runtime_error("FlashAttention only supports fp16/bf16");
            }
        });
    };

    if (head_dim == 256) launch_kernel.template operator()<256>();
    else launch_kernel.template operator()<128>();
    // ===== error check =====
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "flash_attention_warp_kernel launch failed: %s\n",
            cudaGetErrorString(err));
    }
}

template <typename T>
__global__ void paged_attention_kernel(
    T* __restrict__ out,
    const T* __restrict__ Q,
    int32_t head_dim, int32_t num_q_heads, int32_t seq_len,
    int32_t num_kv_heads, int32_t total_cached,
    float scale_val, int causal, int num_kv_groups,
    const int32_t* __restrict__ page_k_ids,
    const int32_t* __restrict__ page_v_ids,
    int32_t num_blocks,
    const T* __restrict__ k_pool_base,
    const T* __restrict__ v_pool_base,
    int32_t block_stride_elems,
    int32_t q_start
) {
    constexpr int32_t MAX_HEAD_DIM = 256;
    constexpr int32_t MAX_HEAD_DIM_REGS = MAX_HEAD_DIM / 32;
    int32_t head_dim_regs = head_dim / 32;

    int32_t bh = blockIdx.x;
    if (bh >= num_q_heads) return;
    int32_t head_idx  = bh % num_q_heads;
    int32_t batch_idx = bh / num_q_heads;
    int32_t kv_head_idx = head_idx / num_kv_groups;

    int32_t warp_id = threadIdx.x / 32;
    int32_t lane_id = threadIdx.x % 32;
    int32_t q_pos = warp_id + blockIdx.y * (blockDim.x / 32);
    if (q_pos >= seq_len) return;

    int64_t q_base = (batch_idx * num_q_heads + head_idx) * seq_len * head_dim;

    float q_reg[MAX_HEAD_DIM_REGS];
    for (int i = 0; i < head_dim_regs; ++i) {
        int d = lane_id + i * 32;
        q_reg[i] = float(Q[q_base + q_pos * head_dim + d]);
    }

    float m_i = -INFINITY;
    float l_i = 0.f;
    float o_reg[MAX_HEAD_DIM_REGS] = {0};

    constexpr int32_t BLOCK_SIZE = PAGE_BLOCK_SIZE;

    for (int32_t blk = 0; blk < num_blocks; ++blk) {
        int32_t k_blk_id = page_k_ids[blk];
        int32_t v_blk_id = page_v_ids[blk];
        if (k_blk_id < 0 || v_blk_id < 0) break;

        const T* k_blk_ptr = k_pool_base + k_blk_id * block_stride_elems;
        const T* v_blk_ptr = v_pool_base + v_blk_id * block_stride_elems;

        int32_t blk_start = blk * BLOCK_SIZE;
        int32_t blk_end = min(blk_start + BLOCK_SIZE, total_cached);

        for (int32_t p = 0; p < blk_end - blk_start; ++p) {
            int32_t kv_pos = blk_start + p;
            if (causal && kv_pos > q_start + q_pos) continue;

            int64_t kv_off = (int64_t(p) * num_kv_heads + kv_head_idx) * head_dim;

            float score_partial = 0.f;
            for (int i = 0; i < head_dim_regs; ++i) {
                int d = lane_id + i * 32;
                score_partial += q_reg[i] * float(k_blk_ptr[kv_off + d]);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                score_partial += __shfl_down_sync(0xffffffff, score_partial, offset);
            float score = __shfl_sync(0xffffffff, score_partial, 0) * scale_val;

            float m_prev = m_i;
            m_i = fmaxf(m_i, score);
            float exp_score = __expf(score - m_i);
            float exp_prev  = __expf(m_prev - m_i);
            for (int i = 0; i < head_dim_regs; ++i) {
                int d = lane_id + i * 32;
                o_reg[i] = o_reg[i] * exp_prev + exp_score * float(v_blk_ptr[kv_off + d]);
            }
            l_i = l_i * exp_prev + exp_score;
        }
    }

    float inv_l = 1.f / l_i;
    for (int i = 0; i < head_dim_regs; ++i) {
        int d = lane_id + i * 32;
        out[q_base + q_pos * head_dim + d] = T(o_reg[i] * inv_l);
    }
}

void PagedAttentionImpl<Device::CUDA>::execute(Tensor* out,int32_t dev_id){
    auto* mgr = g_mem_manager->get_attention_manager(out->device,dev_id);
    int32_t layer_id = out->layer_id;
    if (!mgr || !mgr->is_active(layer_id)) return;

    Tensor* Q = out->src[0];
    Tensor* K = out->src[1];
    Tensor* V = out->src[2];

    int32_t num_q_heads = static_cast<int32_t>(Q->dims[1]);
    int32_t seq_len = static_cast<int32_t>(Q->dims[2]);
    int32_t head_dim = static_cast<int32_t>(Q->dims[3]);

    int32_t num_kv_heads = static_cast<int32_t>(K->dims[1]);
    int32_t num_kv_groups = static_cast<int32_t>(out->op_params[3]);

    float scale = out->op_params[1];
    bool causal = static_cast<int32_t>(out->op_params[2]) != 0;

    if (head_dim % 32 != 0)
        throw std::runtime_error("paged_attention requires head_dim to be a multiple of 32");

    mgr->append_kv_from_tensor(layer_id, K->data, V->data, num_kv_heads, seq_len, head_dim, K->dtype);

    auto& layer = mgr->get_layer(layer_id);
    int32_t total_cached = layer.num_cached;
    int32_t num_blocks = static_cast<int32_t>(layer.page_table.size());
    if (total_cached <= 0 || num_blocks <= 0) return;

    int32_t block_stride_elems = PAGE_BLOCK_SIZE * num_kv_heads * head_dim;

    std::vector<int32_t> h_page_k_ids(num_blocks);
    std::vector<int32_t> h_page_v_ids(num_blocks);
    for (int32_t i = 0; i < num_blocks; ++i) {
        h_page_k_ids[i] = layer.page_table[i].k_block_id;
        h_page_v_ids[i] = layer.page_table[i].v_block_id;
    }

    int32_t *d_page_k_ids = nullptr, *d_page_v_ids = nullptr;
    cudaMalloc(&d_page_k_ids, num_blocks * sizeof(int32_t));
    cudaMalloc(&d_page_v_ids, num_blocks * sizeof(int32_t));
    cudaMemcpy(d_page_k_ids, h_page_k_ids.data(), num_blocks * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_page_v_ids, h_page_v_ids.data(), num_blocks * sizeof(int32_t), cudaMemcpyHostToDevice);

    const void* k_pool_base = layer.k_pool.block_data(0);
    const void* v_pool_base = layer.v_pool.block_data(0);

    constexpr int32_t WARPS_PER_BLOCK = 4;
    constexpr int32_t THREADS = WARPS_PER_BLOCK * 32;
    int32_t blocks_x = num_q_heads;
    int32_t blocks_y = (seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid_dim(blocks_x, blocks_y);

    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        if constexpr (std::is_same_v<T, float16_t>) {
            paged_attention_kernel<__half>
                <<<grid_dim, THREADS>>>(
                    static_cast<__half*>(out->data),
                    static_cast<const __half*>(Q->data),
                    head_dim, num_q_heads, seq_len, num_kv_heads, total_cached,
                    scale, causal, num_kv_groups,
                    d_page_k_ids, d_page_v_ids, num_blocks,
                    static_cast<const __half*>(k_pool_base),
                    static_cast<const __half*>(v_pool_base),
                    block_stride_elems, total_cached - seq_len
                );
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            paged_attention_kernel<__nv_bfloat16>
                <<<grid_dim, THREADS>>>(
                    static_cast<__nv_bfloat16*>(out->data),
                    static_cast<const __nv_bfloat16*>(Q->data),
                    head_dim, num_q_heads, seq_len, num_kv_heads, total_cached,
                    scale, causal, num_kv_groups,
                    d_page_k_ids, d_page_v_ids, num_blocks,
                    static_cast<const __nv_bfloat16*>(k_pool_base),
                    static_cast<const __nv_bfloat16*>(v_pool_base),
                    block_stride_elems, total_cached - seq_len
                );
        }
    });

    cudaFree(d_page_k_ids);
    cudaFree(d_page_v_ids);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "paged_attention_kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template struct PagedAttentionImpl<Device::CUDA>;
template struct FlashAttentionImpl<Device::CUDA>;
}
