#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include "bfloat16.hpp"
#include "float16.hpp"
#include "tools.hpp"
#include "utils/quants.hpp"
#include "backend/cpu/embedding.h"


namespace ops {

template<typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void embedding(
    T* out,
    const T* weight,
    const int32_t* ids,
    int64_t seq_len,
    int64_t hidden,
    int64_t vocab
){
    for (int64_t s = 0; s < seq_len; ++s) {
        int32_t id = ids[s];
        if (id < 0 || id >= vocab) {
            throw 
            std::runtime_error(std::format("cpu::embedding: token id {} out of range [0, {})", id, vocab));
        }
        const T* src_row = weight + id * hidden;
        T* dst_row = out + s * hidden;
        for (int64_t h = 0; h < hidden; ++h)
            dst_row[h] = src_row[h];
    }
}
void embedding(
    bfloat16_t* out,
    const Block_Q4_K* weight,
    const int32_t* ids,
    int64_t seq_len,
    int64_t hidden,
    int64_t vocab
){
    constexpr int QK_K = 256;
    int64_t blocks_per_row = hidden / QK_K;
    for (int64_t s = 0; s < seq_len; ++s) {
        bfloat16_t* dst = out + s * hidden;
        const Block_Q4_K* row_blocks = weight + ids[s] * blocks_per_row; // 先移动指针到对应位置行
        quant::dequantize_row_q4_K(row_blocks,dst,hidden);
    }
}
void embedding(
    bfloat16_t* out,
    const Block_Q6_K* weight,
    const int32_t* ids,
    int64_t seq_len,
    int64_t hidden,
    int64_t vocab
){
    constexpr int QK_K = 256;
    int64_t blocks_per_row = hidden / QK_K;
    for (int64_t s = 0; s < seq_len; ++s) {
        bfloat16_t* dst = out + s * hidden;
        const Block_Q6_K* row_blocks = weight + ids[s] * blocks_per_row; // 先移动指针到对应位置行
        quant::dequantize_row_q6_K(row_blocks,dst,hidden);
    }
}
void embedding(
    bfloat16_t* out,
    const Block_IQ4_NL* weight,
    const int32_t* ids,
    int64_t seq_len,
    int64_t hidden,
    int64_t vocab
){
    int64_t blocks_per_row = hidden / QK4_NL;
    for (int64_t s = 0; s < seq_len; ++s) {
        int32_t id = ids[s];
        if (id < 0 || id >= vocab) {
            throw std::runtime_error(std::format("cpu::embedding: token id {} out of range [0, {})", id, vocab));
        }
        bfloat16_t* dst = out + s * hidden;
        const Block_IQ4_NL* row_blocks = weight + id * blocks_per_row;
        quant::dequantize_row_iq4_nl(row_blocks, dst, hidden);
    }
}
void EmbeddingImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
    const Tensor* input_ids = out->src[0]; // int32_t
    const Tensor* weight    = out->src[1]; // bfloat16_t,Q4_K_M

    if (!input_ids || !input_ids->data)
        throw std::runtime_error("cpu::embedding: input_ids has no data");
    if (!weight || !weight->data)
        throw std::runtime_error("cpu::embedding: weight has no data");

    int64_t vocab = weight->dims[0];
    const int32_t* ids = static_cast<const int32_t*>(input_ids->data);
    int64_t seq = out->dims[1], hidden = out->dims[2];

    // out: 量化类型 -> bf16
    // out: fp16 -> fp16; bf16 -> bf16
    switch (weight->dtype) {
        case DataType::GGML_TYPE_F32:
            embedding<float>(static_cast<float*>(out->data),static_cast<const float*>(weight->data),ids,seq,hidden,vocab);break;
        case DataType::GGML_TYPE_F16: 
            embedding<float16_t>(static_cast<float16_t*>(out->data),static_cast<const float16_t*>(weight->data),ids,seq,hidden,vocab);break;
        case DataType::GGML_TYPE_BF16:
            embedding<bfloat16_t>(static_cast<bfloat16_t*>(out->data),static_cast<const bfloat16_t*>(weight->data),ids,seq,hidden,vocab);break;
        case DataType::GGML_TYPE_Q4_K:
            embedding(static_cast<bfloat16_t*>(out->data),static_cast<const Block_Q4_K*>(weight->data),ids,seq,hidden,vocab);break;
        case DataType::GGML_TYPE_Q6_K:
            embedding(static_cast<bfloat16_t*>(out->data),static_cast<const Block_Q6_K*>(weight->data),ids,seq,hidden,vocab);break;
        case DataType::GGML_TYPE_IQ4_NL:
            embedding(static_cast<bfloat16_t*>(out->data),static_cast<const Block_IQ4_NL*>(weight->data),ids,seq,hidden,vocab);break;
        default: 
            throw std::runtime_error(std::format("embedding: unsupported dtype {}", data_type_to_string(weight->dtype)));
    }
}

template struct EmbeddingImpl<Device::CPU>;
}
