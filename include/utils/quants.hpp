#pragma once
#include <cstdint>
#include "float16.hpp"
#include "bfloat16.hpp"

// 144字节
constexpr int QK_K = 256;
constexpr int QK4_NL = 32;

struct Block_Q4_K {
    float16_t d;            // 超级块的全局缩放因子 (fp16)
    float16_t dmin;         // 超级块的全局最小值因子 (fp16)
    uint8_t scales[12];   // 12字节: 4 + 8
    uint8_t qs[128];      // 128字节，存储256个bfloat16
};

// 128+64+16+2 = 210 bytes, matching GGML block_q6_K layout
struct Block_Q6_K {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    float16_t d;             // super-block scale
};

struct Block_IQ4_NL {
    float16_t d;
    uint8_t qs[QK4_NL / 2];
};

static_assert(sizeof(Block_IQ4_NL) == sizeof(float16_t) + QK4_NL / 2);

namespace quant{
    inline constexpr int8_t kvalues_iq4nl[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };

    inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t*  m) {
        if (j < 4) {
            *d = q[j] & 63; *m = q[j + 4] & 63;
        } else {
            *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
            *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
        }
    }

    inline void dequantize_row_q4_K(const Block_Q4_K* x,bfloat16_t* y,int64_t k){
        const int nb = k / QK_K;
        for (int i = 0; i < nb; i++) {
            const uint8_t * q = x[i].qs;
            const float d   = float(x[i].d);
            const float min = float(x[i].dmin);
            int is = 0;
            uint8_t sc, m;
            for (int j = 0; j < QK_K; j += 64) {
                get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
                const float d1 = d * sc; const float m1 = min * m;
                get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
                const float d2 = d * sc; const float m2 = min * m;
                for (int l = 0; l < 32; ++l) 
                    *y++ = bfloat16_t(d1 * (q[l] & 0xF) - m1);
                for (int l = 0; l < 32; ++l) 
                    *y++ = bfloat16_t(d2 * (q[l]  >> 4) - m2);
                q += 32;
                is += 2;
            }
        }
    }
    inline void dequantize_row_q6_K(const Block_Q6_K * x, bfloat16_t * y, int64_t k) {
        const int64_t nb = k / QK_K;
        for (int i = 0; i < nb; i++) {
            const float d = float(x[i].d);
            const uint8_t *  ql = x[i].ql;
            const uint8_t *  qh = x[i].qh;
            const int8_t  *  sc = x[i].scales;
            for (int n = 0; n < QK_K; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l/16;
                    const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l +  0] = d * sc[is + 0] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y  += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    inline void dequantize_row_iq4_nl(const Block_IQ4_NL* x, bfloat16_t* y, int64_t k) {
        const int64_t nb = k / QK4_NL;
        for (int64_t i = 0; i < nb; ++i) {
            const float d = float(x[i].d);
            const uint8_t* qs = x[i].qs;
            for (int j = 0; j < QK4_NL / 2; ++j) {
                y[j] = bfloat16_t(d * kvalues_iq4nl[qs[j] & 0x0f]);
                y[j + QK4_NL / 2] = bfloat16_t(d * kvalues_iq4nl[qs[j] >> 4]);
            }
            y += QK4_NL;
        }
    }
}
