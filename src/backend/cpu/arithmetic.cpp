#include "backend/cpu/arithmetic.h"
#include "utils/dtype_traits.hpp"

namespace cpu {

// out = src[0] + src[1]  (element-wise)
void add(Tensor* out) {
    const Tensor* a = out->src[0];
    const Tensor* b = out->src[1];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* pa = static_cast<const T*>(a->data);
        const T* pb = static_cast<const T*>(b->data);
        T*       po = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        for (size_t i = 0; i < n; ++i) {
            float fa = dtype::to_f32<D>(pa[i]);
            float fb = dtype::to_f32<D>(pb[i]);
            po[i] = dtype::from_f32<D>(fa + fb);
        }
    });
}

// out = src[0] - src[1]  (element-wise)
void sub(Tensor* out) {
    const Tensor* a = out->src[0];
    const Tensor* b = out->src[1];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* pa = static_cast<const T*>(a->data);
        const T* pb = static_cast<const T*>(b->data);
        T*       po = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        for (size_t i = 0; i < n; ++i) {
            float fa = dtype::to_f32<D>(pa[i]);
            float fb = dtype::to_f32<D>(pb[i]);
            po[i] = dtype::from_f32<D>(fa - fb);
        }
    });
}
// out = src[0] * src[1]  (element-wise / broadcast)
void mul(Tensor* out) {
    const Tensor* a = out->src[0];
    const Tensor* b = out->src[1];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* pa = static_cast<const T*>(a->data);
        const T* pb = static_cast<const T*>(b->data);
        T*       po = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        for (size_t i = 0; i < n; ++i) {
            float fa = dtype::to_f32<D>(pa[i]);
            float fb = dtype::to_f32<D>(pb[i]);
            po[i] = dtype::from_f32<D>(fa * fb);
        }
    });
}
// out = src[0] / src[1]  (element-wise / broadcast)
void div(Tensor* out) {
    const Tensor* a = out->src[0];
    const Tensor* b = out->src[1];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* pa = static_cast<const T*>(a->data);
        const T* pb = static_cast<const T*>(b->data);
        T*       po = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        for (size_t i = 0; i < n; ++i) {
            float fa = dtype::to_f32<D>(pa[i]);
            float fb = dtype::to_f32<D>(pb[i]);
            po[i] = dtype::from_f32<D>(fa / fb);
        }
    });
}
// out = src[0] * op_params[0]  (scalar multiply)
void scale(Tensor* out) {
    const Tensor* x = out->src[0];
    float s = out->op_params[0];
    dtype::dispatch(out->dtype, [&]<DataType D>() {
        using T = dtype::type_t<D>;
        const T* px = static_cast<const T*>(x->data);
        T*       po = static_cast<T*>(out->data);
        size_t   n  = out->num_elements();
        for (size_t i = 0; i < n; ++i) {
            po[i] = dtype::from_f32<D>(dtype::to_f32<D>(px[i]) * s);
        }
    });
}

} // namespace cpu
