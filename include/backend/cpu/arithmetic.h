#pragma once
#include "core/tensor.hpp"

namespace cpu {

// out = src[0] + src[1]  (element-wise)
void add(Tensor* out);

// out = src[0] - src[1]  (element-wise)
void sub(Tensor* out);

// out = src[0] * src[1]  (element-wise / broadcast)
void mul(Tensor* out);

// out = src[0] / src[1]  (element-wise / broadcast)
void div(Tensor* out);

// out = src[0] * op_params[0]  (scalar multiply)
void scale(Tensor* out);

} // namespace cpu
