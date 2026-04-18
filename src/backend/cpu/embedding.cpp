#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>
#include "backend/cpu/embedding.h"
#include "utils/utils.hpp"

namespace cpu {

// transpose=false: weight [vocab, hidden]，直接行拷贝
// transpose=true:  weight [hidden, vocab]，需要逐行 gather 第 id 列
void embedding(Tensor* out) {
    const Tensor* input_ids = out->src[0];
    const Tensor* weight     = out->src[1];
    bool transpose = out->op_params[0] == 1;

    if (!input_ids || !input_ids->data) {
        throw std::runtime_error(std::format("cpu::embedding: input_ids '{}' has no data",
            input_ids ? input_ids->name : "null"));
    }
    if (!weight || !weight->data) {
        throw std::runtime_error(std::format("cpu::embedding: weight '{}' has no data",
            weight ? weight->name : "null"));
    }

    const int32_t* ids = static_cast<const int32_t*>(input_ids->data);
    auto* dst = static_cast<uint8_t*>(out->data);
    const auto* w = static_cast<const uint8_t*>(weight->data);

    const int64_t batch  = out->dims[0];
    const int64_t seq    = out->dims[1];
    const int64_t hidden = out->dims[2];

    size_t elem_sz = data_type_size(weight->dtype);
    size_t out_row_bytes = static_cast<size_t>(hidden) * elem_sz;

    if (!transpose) {
        // weight [vocab, hidden]: token id 直接对应行号
        const int64_t vocab = weight->dims[0];
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                int32_t id = ids[b * seq + s];
                if (id < 0 || id >= vocab) {
                    throw std::runtime_error(std::format(
                        "cpu::embedding: token id {} out of range [0, {})", id, vocab));
                }
                std::memcpy(
                    dst + (b * seq + s) * out_row_bytes,
                    w + static_cast<size_t>(id) * out_row_bytes,
                    out_row_bytes
                );
            }
        }
    } else {
        // weight [hidden, vocab]: 从每行 gather 第 id 列
        const int64_t vocab = weight->dims[1];
        size_t w_row_bytes = static_cast<size_t>(vocab) * elem_sz;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                int32_t id = ids[b * seq + s];
                if (id < 0 || id >= vocab) {
                    throw std::runtime_error(std::format(
                        "cpu::embedding: token id {} out of range [0, {})", id, vocab));
                }
                auto* dst_row = dst + (b * seq + s) * out_row_bytes;
                for (int64_t h = 0; h < hidden; ++h) {
                    std::memcpy(
                        dst_row + h * elem_sz,
                        w + h * w_row_bytes + static_cast<size_t>(id) * elem_sz,
                        elem_sz
                    );
                }
            }
        }
    }
}

} // namespace cpu
