#include "core/kernels.h"
#include <format>
#include <stdexcept>

#define STUB_KERNEL(kfn) \
    void kfn(Tensor* t) { \
        throw std::runtime_error(std::format("kernel::{} not implemented (tensor: {})", #kfn, t->name)); \
    }

namespace kernel {
    STUB_KERNEL(add)
    STUB_KERNEL(sub)
    STUB_KERNEL(mul)
    STUB_KERNEL(div)
    STUB_KERNEL(scale)

    STUB_KERNEL(rms_norm)
    STUB_KERNEL(layer_norm)

    STUB_KERNEL(matmul)
    STUB_KERNEL(linear)
    STUB_KERNEL(transpose)

    STUB_KERNEL(reshape)
    STUB_KERNEL(permute)

    STUB_KERNEL(silu)
    STUB_KERNEL(gelu)
    STUB_KERNEL(relu)

    STUB_KERNEL(softmax)
    STUB_KERNEL(diag_mask_inf)

    STUB_KERNEL(embedding)
    STUB_KERNEL(apply_rope)
    STUB_KERNEL(sdpa)
    STUB_KERNEL(flash_attn)

    STUB_KERNEL(get_rows)
    STUB_KERNEL(concat)
    STUB_KERNEL(repeat)

    STUB_KERNEL(rope_cache)
    STUB_KERNEL(sampling)
}
