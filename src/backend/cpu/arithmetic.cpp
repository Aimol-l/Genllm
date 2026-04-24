#include "backend/cpu/arithmetic.h"
#include "utils/dtype_traits.hpp"

namespace ops {

    void AddImpl<Device::CPU>::execute(Tensor* out) {
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

    void SubImpl<Device::CPU>::execute(Tensor* out) {
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

    void MulImpl<Device::CPU>::execute(Tensor* out) {
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

    void DivImpl<Device::CPU>::execute(Tensor* out) {
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

template struct AddImpl<Device::CPU>;
template struct SubImpl<Device::CPU>;
template struct MulImpl<Device::CPU>;
template struct DivImpl<Device::CPU>;
}
