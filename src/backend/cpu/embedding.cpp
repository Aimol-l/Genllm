#include <format>
#include <stdexcept>
#include "backend/cpu/embedding.h"
#include "utils/dtype_traits.hpp"

// transpose=false: weight [vocab, hidden]，直接行拷贝
// transpose=true:  weight [hidden, vocab]，逐列 gather
template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void embedding(
    T* out,
    const int32_t* ids,
    const T* weight,
    int64_t batch,
    int64_t seq,
    int64_t hidden,
    int64_t vocab,
    bool transpose
){
    if (!transpose) {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                int32_t id = ids[b * seq + s];
                if (id < 0 || id >= vocab) {
                    throw std::runtime_error(std::format(
                        "cpu::embedding: token id {} out of range [0, {})", id, vocab));
                }
                const T* src_row = weight + id * hidden;
                T* dst_row = out + (b * seq + s) * hidden;
                for (int64_t h = 0; h < hidden; ++h)
                    dst_row[h] = src_row[h];
            }
        }
    } else {
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                int32_t id = ids[b * seq + s];
                if (id < 0 || id >= vocab) {
                    throw std::runtime_error(std::format(
                        "cpu::embedding: token id {} out of range [0, {})", id, vocab));
                }
                T* dst_row = out + (b * seq + s) * hidden;
                for (int64_t h = 0; h < hidden; ++h)
                    dst_row[h] = weight[h * vocab + id];
            }
        }
    }
}

namespace ops {

    void EmbeddingImpl<Device::CPU>::execute(Tensor* out) {
        const Tensor* input_ids = out->src[0];
        const Tensor* weight = out->src[1];
        bool transpose = out->op_params[0] == 1;

        if (!input_ids || !input_ids->data)
            throw std::runtime_error("cpu::embedding: input_ids has no data");
        if (!weight || !weight->data)
            throw std::runtime_error("cpu::embedding: weight has no data");

        const int32_t* ids = static_cast<const int32_t*>(input_ids->data);
        int64_t batch = out->dims[0], seq = out->dims[1], hidden = out->dims[2];
        int64_t vocab = transpose ? weight->dims[1] : weight->dims[0];

        dtype::dispatch(weight->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            embedding(static_cast<T*>(out->data), ids, static_cast<const T*>(weight->data),batch, seq, hidden, vocab, transpose);
        });
    }

template struct EmbeddingImpl<Device::CPU>;
}
