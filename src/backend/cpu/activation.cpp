#include <cmath>
#include "backend/cpu/activation.h"
#include "utils/dtype_traits.hpp"

namespace cpu {

// out = x * sigmoid(x)
void silu(Tensor* out) {
    const Tensor* x = out->src[0];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* in = static_cast<const T*>(x->data);
        T*       o  = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        for (size_t i = 0; i < n; ++i) {
            float fx = dtype::to_f32<D>(in[i]);
            o[i] = dtype::from_f32<D>(fx / (1.0f + std::exp(-fx)));
        }
    });
}

// out = x * 0.5 * (1 + erf(x / sqrt(2)))
void gelu(Tensor* out) {
    const Tensor* x = out->src[0];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* in = static_cast<const T*>(x->data);
        T*       o  = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        constexpr float inv_sqrt2 = 0.7071067811865475f;
        for (size_t i = 0; i < n; ++i) {
            float fx = dtype::to_f32<D>(in[i]);
            o[i] = dtype::from_f32<D>(fx * 0.5f * (1.0f + std::erf(fx * inv_sqrt2)));
        }
    });
}
// out = max(0, x)
void relu(Tensor* out) {
    const Tensor* x = out->src[0];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* in = static_cast<const T*>(x->data);
        T*       o  = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        for (size_t i = 0; i < n; ++i) {
            float fx = dtype::to_f32<D>(in[i]);
            o[i] = dtype::from_f32<D>(fx > 0.0f ? fx : 0.0f);
        }
    });
}

} // namespace cpu
