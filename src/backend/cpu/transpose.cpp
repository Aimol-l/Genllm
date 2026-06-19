#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "utils/bfloat16.hpp"
#include "utils/float16.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cpu/transpose.h"

template <typename T> requires std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float> || std::is_same_v<T, float16_t>
void transpose(
    T* out,
    const T* in,
    const int64_t* src_dims,
    const uint64_t* src_strides,
    int ndim,
    size_t total
){
    for (size_t idx = 0; idx < total; ++idx) {
        size_t remaining  = idx;
        size_t src_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            size_t dim_size = static_cast<size_t>(src_dims[d]);
            size_t coord    = remaining % dim_size;
            remaining /= dim_size;
            src_offset += coord * src_strides[d];
        }
        out[idx] = in[src_offset];
    }
}
namespace ops {
    void TransposeImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        int64_t ax0 = static_cast<int64_t>(out->op_params[0]);
        int64_t ax1 = static_cast<int64_t>(out->op_params[1]);

        int ndim = 0;
        for (int i = 0; i < TENSOR_MAX_DIMS && x->dims[i] != 0; ++i) {
            ndim = i + 1;
        }

        int64_t src_dims[TENSOR_MAX_DIMS]{};
        uint64_t src_strides[TENSOR_MAX_DIMS]{};
        for (int i = 0; i < ndim; ++i) {
            src_dims[i]    = x->dims[i];
            src_strides[i] = x->strides[i];
        }
        std::swap(src_dims[ax0], src_dims[ax1]);
        std::swap(src_strides[ax0], src_strides[ax1]);

        dtype::dispatch(x->dtype, [&]<DataType D_x>() {
            using Tx = dtype::type_t<D_x>;
            auto* op = static_cast<Tx*>(out->data);
            auto* xp = static_cast<const Tx*>(x->data);
            transpose<Tx>(op, xp, src_dims, src_strides, ndim, out->num_elements());
        });
    }

template struct TransposeImpl<Device::CPU>;
}