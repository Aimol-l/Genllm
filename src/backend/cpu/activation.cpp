#include <cmath>
#include "utils/dtype_traits.hpp"
#include "backend/cpu/activation.h"

namespace ops {

    void SiluImpl<Device::CPU>::execute(Tensor* out) {
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

    void GeluImpl<Device::CPU>::execute(Tensor* out) {
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

    void ReluImpl<Device::CPU>::execute(Tensor* out) {
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

template struct SiluImpl<Device::CPU>;
template struct GeluImpl<Device::CPU>;
template struct ReluImpl<Device::CPU>;
}
