#pragma once
#include "core/tensor.hpp"

namespace ops {

struct LinearPushConstants {
    int32_t M, K, N;
};

struct NormPushConstants {
    int32_t rows;
    int32_t hidden_size;
    float eps;
};

struct LayerNormPushConstants {
    int32_t rows;
    int32_t hidden_size;
    float eps;
    int32_t has_bias;
};

struct RopePushConstants {
    int32_t head_dim;
    int32_t half_dim;
    int32_t seq_len;
    int32_t start_pos;
    int32_t n_heads;
    int32_t B;
};

struct PermutePushConstants {
    int32_t ndim;
    int32_t total;
    int32_t out_strides[TENSOR_MAX_DIMS];
    int32_t src_strides[TENSOR_MAX_DIMS];
    int32_t perm[TENSOR_MAX_DIMS];
};

struct FlashPushConstants {
    int32_t batch;
    int32_t seq_len;
    int32_t head_dim;
    int32_t num_q_heads;
    int32_t num_kv_heads;
    int32_t group_size;  // = num_q_heads / num_kv_heads
    float   scale;       // = 1.0f / sqrtf(head_dim)
};

struct PagedAttnPushConstants {
    int32_t batch;
    int32_t seq_len;
    int32_t num_q_heads;
    int32_t num_kv_heads;
    int32_t total_cached;
    int32_t num_blocks;
    int32_t num_kv_groups;
    int32_t head_dim;
    int32_t block_stride_elems;
    int32_t q_start;
    float   scale;      // = 1.0f / sqrtf(head_dim)
};

} // namespace ops
