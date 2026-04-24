#include "core/kernels.h"
#include "utils.hpp"
#include "utils/dtype_traits.hpp"
#include "backend/cpu/arithmetic.h"
#include "backend/cpu/normalization.h"
#include "backend/cpu/linear.h"
#include "backend/cpu/shape.h"
#include "backend/cpu/activation.h"
#include "backend/cpu/attention.h"
#include "backend/cpu/embedding.h"
#include "backend/cpu/rope.h"
#ifdef BACKEND_CUDA
#include "backend/cuda/arithmetic.h"
#include "backend/cuda/normalization.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/shape.h"
#include "backend/cuda/activation.h"
#include "backend/cuda/attention.h"
#include "backend/cuda/rope.h"
#endif

namespace kernel {

    // ===== arithmetic =====
    void add(Tensor* t)       { device::dispatchOp(t->device, [&]<Device D>() { ops::AddImpl<D>::execute(t); }); }
    void sub(Tensor* t)       { device::dispatchOp(t->device, [&]<Device D>() { ops::SubImpl<D>::execute(t); }); }
    void mul(Tensor* t)       { device::dispatchOp(t->device, [&]<Device D>() { ops::MulImpl<D>::execute(t); }); }
    void div(Tensor* t)       { device::dispatchOp(t->device, [&]<Device D>() { ops::DivImpl<D>::execute(t); }); }

    // ===== normalization =====
    void rms_norm(Tensor* t)  { device::dispatchOp(t->device, [&]<Device D>() { ops::RmsNormImpl<D>::execute(t); }); }
    void layer_norm(Tensor* t){ device::dispatchOp(t->device, [&]<Device D>() { ops::LayerNormImpl<D>::execute(t); }); }

    // ===== linear / matmul =====
    void matmul(Tensor* t)    { device::dispatchOp(t->device, [&]<Device D>() { ops::MatmulImpl<D>::execute(t); }); }
    void linear(Tensor* t)    { device::dispatchOp(t->device, [&]<Device D>() { ops::LinearImpl<D>::execute(t); }); }
    void transpose(Tensor* t) { device::dispatchOp(t->device, [&]<Device D>() { ops::TransposeImpl<D>::execute(t); }); }

    // ===== shape =====
    void reshape(Tensor* t)   { device::dispatchOp(t->device, [&]<Device D>() { ops::ReshapeImpl<D>::execute(t); }); }
    void permute(Tensor* t)   { device::dispatchOp(t->device, [&]<Device D>() { ops::PermuteImpl<D>::execute(t); }); }

    // ===== activation =====
    void silu(Tensor* t)      { device::dispatchOp(t->device, [&]<Device D>() { ops::SiluImpl<D>::execute(t); }); }
    void gelu(Tensor* t)      { device::dispatchOp(t->device, [&]<Device D>() { ops::GeluImpl<D>::execute(t); }); }
    void relu(Tensor* t)      { device::dispatchOp(t->device, [&]<Device D>() { ops::ReluImpl<D>::execute(t); }); }

    // ===== attention =====
    void softmax(Tensor* t)        { device::dispatchOp(t->device, [&]<Device D>() { ops::SoftmaxImpl<D>::execute(t); }); }
    void diag_mask_inf(Tensor* t)  { device::dispatchOp(t->device, [&]<Device D>() { ops::DiagMaskInfImpl<D>::execute(t); }); }
    void sdpa(Tensor* t)           { device::dispatchOp(t->device, [&]<Device D>() { ops::SdpaImpl<D>::execute(t); }); }
    void attention(Tensor* t)      { device::dispatchOp(t->device, [&]<Device D>() { ops::AttentionImpl<D>::execute(t); }); }
    void flash_attention(Tensor* t){ device::dispatchOp(t->device, [&]<Device D>() { ops::FlashAttentionImpl<D>::execute(t); }); }

    // ===== embedding =====
    void embedding(Tensor* t){
        ops::EmbeddingImpl<Device::CPU>::execute(t);
    }

    // ===== rope =====
    void apply_rope(Tensor* t)     { device::dispatchOp(t->device, [&]<Device D>() { ops::ApplyRopeImpl<D>::execute(t); }); }
    void rope_cache(Tensor* t){
        ops::RopeCacheImpl<Device::CPU>::execute(t);
    }

    // ===== misc =====
    void concat(Tensor* t)         { device::dispatchOp(t->device, [&]<Device D>() { ops::ConcatImpl<D>::execute(t); }); }
    void repeat(Tensor* t)         { device::dispatchOp(t->device, [&]<Device D>() { ops::RepeatImpl<D>::execute(t); }); }
} // namespace kernel
