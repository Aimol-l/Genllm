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
#include "backend/cpu/memcpy.h"
#ifdef BACKEND_CUDA
#include "backend/cuda/arithmetic.h"
#include "backend/cuda/normalization.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/shape.h"
#include "backend/cuda/activation.h"
#include "backend/cuda/attention.h"
#include "backend/cuda/rope.h"
#include "backend/cuda/memcpy.h"
#include <cuda_runtime.h>
#endif

namespace {
#ifdef BACKEND_CUDA
    void set_device_for_op(Device dev, int32_t) {
    }
#else
    void set_device_for_op(Device, int32_t) {} // no-op,估计会被编译器优化掉
#endif
} // anonymous namespace

namespace kernel {

    // ===== arithmetic =====
    void add(Tensor* t, int32_t dev_id)       { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::AddImpl<D>::execute(t); }); }
    void sub(Tensor* t, int32_t dev_id)       { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::SubImpl<D>::execute(t); }); }
    void mul(Tensor* t, int32_t dev_id)       { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::MulImpl<D>::execute(t); }); }
    void div(Tensor* t, int32_t dev_id)       { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::DivImpl<D>::execute(t); }); }

    // ===== normalization =====
    void rms_norm(Tensor* t, int32_t dev_id)  { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::RmsNormImpl<D>::execute(t); }); }
    void layer_norm(Tensor* t, int32_t dev_id){ set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::LayerNormImpl<D>::execute(t); }); }

    // ===== linear / matmul =====
    void matmul(Tensor* t, int32_t dev_id)    { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::MatmulImpl<D>::execute(t); }); }
    void linear(Tensor* t, int32_t dev_id)    { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::LinearImpl<D>::execute(t); }); }
    void transpose(Tensor* t, int32_t dev_id) { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::TransposeImpl<D>::execute(t); }); }

    // ===== shape =====
    void reshape(Tensor* t, int32_t dev_id)   { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::ReshapeImpl<D>::execute(t); }); }
    void permute(Tensor* t, int32_t dev_id)   { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::PermuteImpl<D>::execute(t); }); }

    // ===== activation =====
    void silu(Tensor* t, int32_t dev_id)      { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::SiluImpl<D>::execute(t); }); }
    void gelu(Tensor* t, int32_t dev_id)      { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::GeluImpl<D>::execute(t); }); }
    void relu(Tensor* t, int32_t dev_id)      { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::ReluImpl<D>::execute(t); }); }

    // ===== attention =====
    void softmax(Tensor* t, int32_t dev_id)        { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::SoftmaxImpl<D>::execute(t); }); }
    void diag_mask_inf(Tensor* t, int32_t dev_id)  { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::DiagMaskInfImpl<D>::execute(t); }); }
    void sdpa(Tensor* t, int32_t dev_id)           { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::SdpaImpl<D>::execute(t, dev_id); }); }
    void attention(Tensor* t, int32_t dev_id)      { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::AttentionImpl<D>::execute(t); }); }
    void flash_attention(Tensor* t, int32_t dev_id){ set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::FlashAttentionImpl<D>::execute(t); }); }

    // ===== embedding =====
    void embedding(Tensor* t, int32_t dev_id){
        set_device_for_op(t->device, dev_id);
        ops::EmbeddingImpl<Device::CPU>::execute(t);
    }

    // ===== rope =====
    void apply_rope(Tensor* t, int32_t dev_id)     { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::ApplyRopeImpl<D>::execute(t); }); }
    void rope_cache(Tensor* t, int32_t dev_id){
        set_device_for_op(t->device, dev_id);
        ops::RopeCacheImpl<Device::CPU>::execute(t);
    }

    // ===== memcpy =====
    void memcpy(Tensor* t, int32_t dev_id) {
        Device dev = t->device;
        if (t->src[0] && t->src[0]->device != t->device) {
            dev = (t->device != Device::CPU) ? t->device : t->src[0]->device;
        }
        set_device_for_op(dev, dev_id);
        device::dispatchOp(dev, [&]<Device D>() { ops::MemcpyImpl<D>::execute(t); });
    }

    // ===== misc =====
    void concat(Tensor* t, int32_t dev_id)         { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::ConcatImpl<D>::execute(t); }); }
    void repeat(Tensor* t, int32_t dev_id)         { set_device_for_op(t->device, dev_id); device::dispatchOp(t->device, [&]<Device D>() { ops::RepeatImpl<D>::execute(t); }); }
} // namespace kernel
