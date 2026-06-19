#include <cmath>
#include <limits>
#include "utils/dtype_traits.hpp"
#include "backend/cpu/softmax.h"

namespace ops {

    void SoftmaxImpl<Device::CPU>::execute(Tensor* out, int32_t dev_id) {
        const Tensor* x = out->src[0];
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            const T* px = static_cast<const T*>(x->data);
            T*       po = static_cast<T*>(out->data);
            size_t   n  = out->num_elements();

            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < n; ++i) {
                float fx = dtype::to_f32<D>(px[i]);
                if (fx > max_val) max_val = fx;
            }

            float sum = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                float fx = std::exp(dtype::to_f32<D>(px[i]) - max_val);
                sum += fx;
                po[i] = dtype::from_f32<D>(fx);
            }

            for (size_t i = 0; i < n; ++i) {
                po[i] = dtype::from_f32<D>(dtype::to_f32<D>(po[i]) / sum);
            }
        });
    }

template struct SoftmaxImpl<Device::CPU>;
}
