#include "utils/dtype_traits.hpp"
#include "utils/float16.hpp"
#include "utils/bfloat16.hpp"
#include <cstdint>
#include <cmath>
#include <cstring>
#include "backend/cpu/mamba.h"

// conv_state 布局: [kernel_size-1, D_inner] = [3, w_len]
//   行 0 = 最老, 行 2 = 最新
//   prefill 结束后保存 input 的最后 3 行
//   decode 时作为左侧 padding 供滑动窗口使用

template<typename T>
void causal_conv1d(
    T* __restrict__ out,           // [1, seq, D_inner]
    T* __restrict__ input,         // [1, seq, D_inner]
    float* __restrict__ weight,    // [D_inner, kernel_size]
    T* __restrict__ conv_state,    // [kernel_size-1, D_inner]
    int32_t k_size, int32_t seq_len, int32_t w_len)
{
    int32_t pad = k_size - 1;
    for (int32_t t = 0; t < seq_len; t++) {
        T* out_row = out + t * w_len;
        for (int32_t d = 0; d < w_len; d++) {
            float acc = 0.0f;
            for (int32_t k = 0; k < k_size; k++) {
                int32_t src_t = t - k;
                float x_val;
                if (src_t >= 0) {
                    x_val = static_cast<float>(input[src_t * w_len + d]);
                } else {
                    int32_t state_row = src_t + pad;
                    x_val = static_cast<float>(conv_state[state_row * w_len + d]);
                }
                acc += weight[d * k_size + k] * x_val;
            }
            out_row[d] = static_cast<T>(acc);
        }
    }
    // 更新 conv_state: 保留 input 的最后 pad 行
    if (seq_len >= pad) {
        std::memcpy(conv_state, input + (seq_len - pad) * w_len, pad * w_len * sizeof(T));
    } else {
        // seq_len < pad: 老状态左移，再追加新输入
        std::memmove(conv_state, conv_state + seq_len * w_len, (pad - seq_len) * w_len * sizeof(T));
        std::memcpy(conv_state + (pad - seq_len) * w_len, input, seq_len * w_len * sizeof(T));
    }
}

// ──────────────────────────────────────────────────────────────────
// Gated Delta Rule (Qwen3.5 SSM layer)
// 参考: HuggingFace torch_recurrent_gated_delta_rule
//
// 对每个时间步 t, 每个 head h:
//   g     = exp(-exp(A_log[h]) * softplus(a[t,h] + dt_bias[h]))
//   beta  = sigmoid(b[t,h])
//   state = state * g                           (衰减)
//   kv_mem[j] = sum_i state[i,j] * K[i]         (回忆)
//   delta[j]  = (V[j] - kv_mem[j]) * beta       (误差)
//   state[i,j] += K[i] * delta[j]               (写入)
//   y[j] = sum_i state[i,j] * Q[i]              (检索)
//
// Q, K 做 L2 归一化, Q 额外乘 1/sqrt(D)
// ──────────────────────────────────────────────────────────────────

static inline float ssm_softplus(float x) {
    if (x > 20.0f) return x;
    return std::log1pf(std::expf(x));
}

static inline float ssm_sigmoid(float x) {
    return 1.0f / (1.0f + std::expf(-x));
}

constexpr int32_t SSM_MAX_HEAD_DIM = 256;

template<typename T>
void gated_delta_rule_ref(
    T* __restrict__ out,               // [seq, value_dim]
    const T* __restrict__ input,       // [seq, conv_dim] (Q+K+V after silu)
    const float* __restrict__ a_log,   // [num_heads] A_log
    const T* __restrict__ b_data,      // [seq, num_heads] beta logit
    const T* __restrict__ a_data,      // [seq, num_heads] alpha/dt
    const float* __restrict__ dt_bias, // [num_heads]
    float* __restrict__ state,         // [num_heads, head_dim, head_dim]
    int32_t seq_len, int32_t conv_dim, int32_t num_heads, int32_t head_dim)
{
    if (head_dim > SSM_MAX_HEAD_DIM) {
        throw std::runtime_error(std::format("gated_delta_rule: head_dim={} exceeds SSM_MAX_HEAD_DIM={}", head_dim, SSM_MAX_HEAD_DIM));
    }
    const int32_t key_dim = (conv_dim - num_heads * head_dim) / 2;
    const int32_t value_dim = num_heads * head_dim;
    const int32_t D = head_dim;
    const int32_t DD = D * D;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(D));
    for (int32_t t = 0; t < seq_len; t++) {
        const T* qkv = input + static_cast<int64_t>(t) * conv_dim;
        T* out_row = out + static_cast<int64_t>(t) * value_dim;
        for (int32_t h = 0; h < num_heads; h++) {
            float* s = state + h * DD;
            float b_val = static_cast<float>(b_data[static_cast<int64_t>(t) * num_heads + h]);
            float a_val = static_cast<float>(a_data[static_cast<int64_t>(t) * num_heads + h]);
            float beta = ssm_sigmoid(b_val);
            float g = std::expf(-std::expf(a_log[h]) * ssm_softplus(a_val + dt_bias[h]));
            const T* q_raw = qkv + h * D;
            const T* k_raw = qkv + key_dim + h * D;
            const T* v_raw = qkv + 2 * key_dim + h * D;

            float q_sq = 0.0f, k_sq = 0.0f;
            float q_buf[SSM_MAX_HEAD_DIM], k_buf[SSM_MAX_HEAD_DIM], v_buf[SSM_MAX_HEAD_DIM];
            for (int32_t d = 0; d < D; d++) {
                q_buf[d] = static_cast<float>(q_raw[d]);
                k_buf[d] = static_cast<float>(k_raw[d]);
                v_buf[d] = static_cast<float>(v_raw[d]);
                q_sq += q_buf[d] * q_buf[d];
                k_sq += k_buf[d] * k_buf[d];
            }
            float q_inv = q_scale / std::sqrt(q_sq + 1e-6f);
            float k_inv = 1.0f / std::sqrt(k_sq + 1e-6f);
            for (int32_t d = 0; d < D; d++) {
                q_buf[d] *= q_inv;
                k_buf[d] *= k_inv;
            }
            for (int32_t i = 0; i < DD; i++) {
                s[i] *= g;
            }
            float kv_mem[SSM_MAX_HEAD_DIM];
            for (int32_t j = 0; j < D; j++) {
                float sum = 0.0f;
                for (int32_t i = 0; i < D; i++) {
                    sum += s[i * D + j] * k_buf[i];
                }
                kv_mem[j] = sum;
            }
            float delta[SSM_MAX_HEAD_DIM];
            for (int32_t j = 0; j < D; j++) {
                delta[j] = (v_buf[j] - kv_mem[j]) * beta;
            }
            for (int32_t i = 0; i < D; i++) {
                float ki = k_buf[i];
                for (int32_t j = 0; j < D; j++) {
                    s[i * D + j] += ki * delta[j];
                }
            }
            for (int32_t j = 0; j < D; j++) {
                float sum = 0.0f;
                for (int32_t i = 0; i < D; i++) {
                    sum += s[i * D + j] * q_buf[i];
                }
                out_row[h * D + j] = static_cast<T>(sum);
            }
        }
    }
}

namespace ops {

void CausalConv1dImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
    Tensor* input      = out->src[0];   // [1, seq, D_inner]
    Tensor* weight     = out->src[1];   // [D_inner, kernel_size]
    Tensor* conv_state = out->src[2];   // [1, kernel_size-1, D_inner] (CACHE)

    int32_t seq_len = static_cast<int32_t>(input->dims[1]);
    int32_t w_len   = static_cast<int32_t>(weight->dims[0]);    // 6144
    int32_t k_size  = static_cast<int32_t>(out->op_params[0]);  // 4

    dtype::dispatch(out->dtype, [&]<DataType D_w>() {
        using T = dtype::type_t<D_w>;
        causal_conv1d<T>(
            static_cast<T*>(out->data),
            static_cast<T*>(input->data),
            static_cast<float*>(weight->data),
            static_cast<T*>(conv_state->data),
            k_size, seq_len, w_len
        );
    });
}

void SsmScanImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
    Tensor* input     = out->src[0];   // conv_act [1, seq, conv_dim]
    Tensor* ssm_a     = out->src[1];   // A_log [num_heads] F32
    Tensor* b_proj    = out->src[2];   // b [1, seq, num_heads]
    Tensor* a_proj    = out->src[3];   // a [1, seq, num_heads]
    Tensor* ssm_dt    = out->src[4];   // dt_bias [num_heads] F32
    Tensor* ssm_state = out->src[5];   // state [1, num_heads, D, D] F32

    int32_t num_heads = static_cast<int32_t>(out->op_params[0]);
    int32_t head_dim  = static_cast<int32_t>(out->op_params[1]);
    int32_t seq_len   = static_cast<int32_t>(input->dims[1]);
    int32_t conv_dim  = static_cast<int32_t>(input->dims[2]);

    dtype::dispatch(out->dtype, [&]<DataType D_w>() {
        using T = dtype::type_t<D_w>;
        gated_delta_rule_ref<T>(
            static_cast<T*>(out->data),
            static_cast<const T*>(input->data),
            static_cast<const float*>(ssm_a->data),
            static_cast<const T*>(b_proj->data),
            static_cast<const T*>(a_proj->data),
            static_cast<const float*>(ssm_dt->data),
            static_cast<float*>(ssm_state->data),
            seq_len, conv_dim, num_heads, head_dim
        );
    });
}

template struct CausalConv1dImpl<Device::CPU>;
template struct SsmScanImpl<Device::CPU>;

}
