#include "utils.hpp"
#include "core/kernels.h"

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
#include "backend/cpu/mamba.h"
#include "backend/cpu/softmax.h"
#include "backend/cpu/transpose.h"
#ifdef BACKEND_CUDA
#include "backend/cuda/arithmetic.h"
#include "backend/cuda/normalization.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/shape.h"
#include "backend/cuda/activation.h"
#include "backend/cuda/attention.h"
#include "backend/cuda/rope.h"
#include "backend/cuda/memcpy.h"
#include "backend/cuda/mamba.h"
#include "backend/cuda/softmax.h"
#include "backend/cuda/transpose.h"

#include <cuda_runtime.h>
#endif
#ifdef BACKEND_VULKAN
#include "backend/vulkan/arithmetic.h"
#include "backend/vulkan/normalization.h"
#include "backend/vulkan/linear.h"
#include "backend/vulkan/shape.h"
#include "backend/vulkan/activation.h"
#include "backend/vulkan/attention.h"
#include "backend/vulkan/rope.h"
#include "backend/vulkan/memcpy.h"
#include "backend/vulkan/mamba.h"
#include "backend/vulkan/softmax.h"
#include "backend/vulkan/transpose.h"
#endif

namespace kernel {

    // ===== arithmetic =====
    void add(Tensor* t, int32_t dev_id)       { device::dispatchOp(t->device, [&]<Device D>() { ops::AddImpl<D>::execute(t, dev_id); }); }
    void sub(Tensor* t, int32_t dev_id)       { device::dispatchOp(t->device, [&]<Device D>() { ops::SubImpl<D>::execute(t, dev_id); }); }
    void mul(Tensor* t, int32_t dev_id)       { device::dispatchOp(t->device, [&]<Device D>() { ops::MulImpl<D>::execute(t, dev_id); }); }
    void div(Tensor* t, int32_t dev_id)       { device::dispatchOp(t->device, [&]<Device D>() { ops::DivImpl<D>::execute(t, dev_id); }); }

    // ===== normalization =====
    void rms_norm(Tensor* t, int32_t dev_id)  { device::dispatchOp(t->device, [&]<Device D>() { ops::RmsNormImpl<D>::execute(t, dev_id); }); }
    void layer_norm(Tensor* t, int32_t dev_id){ device::dispatchOp(t->device, [&]<Device D>() { ops::LayerNormImpl<D>::execute(t, dev_id); }); }

    // ===== linear / matmul =====
    void matmul(Tensor* t, int32_t dev_id)    { device::dispatchOp(t->device, [&]<Device D>() { ops::MatmulImpl<D>::execute(t, dev_id); }); }
    void linear(Tensor* t, int32_t dev_id)    { device::dispatchOp(t->device, [&]<Device D>() { ops::LinearImpl<D>::execute(t, dev_id); }); }
    void transpose(Tensor* t, int32_t dev_id) { device::dispatchOp(t->device, [&]<Device D>() { ops::TransposeImpl<D>::execute(t, dev_id); }); }

    // ===== shape =====
    void reshape(Tensor* t, int32_t dev_id)   { device::dispatchOp(t->device, [&]<Device D>() { ops::ReshapeImpl<D>::execute(t, dev_id); }); }
    void permute(Tensor* t, int32_t dev_id)   { device::dispatchOp(t->device, [&]<Device D>() { ops::PermuteImpl<D>::execute(t, dev_id); }); }

    // ===== activation =====
    void silu(Tensor* t, int32_t dev_id)      { device::dispatchOp(t->device, [&]<Device D>() { ops::SiluImpl<D>::execute(t, dev_id); }); }
    void gelu(Tensor* t, int32_t dev_id)      { device::dispatchOp(t->device, [&]<Device D>() { ops::GeluImpl<D>::execute(t, dev_id); }); }
    void relu(Tensor* t, int32_t dev_id)      { device::dispatchOp(t->device, [&]<Device D>() { ops::ReluImpl<D>::execute(t, dev_id); }); }
    void sigmoid(Tensor* t, int32_t dev_id)   { device::dispatchOp(t->device, [&]<Device D>() { ops::SigmoidImpl<D>::execute(t, dev_id); }); }

    // ===== attention =====
    void softmax(Tensor* t, int32_t dev_id)        { device::dispatchOp(t->device, [&]<Device D>() { ops::SoftmaxImpl<D>::execute(t, dev_id); }); }
    void paged_attention(Tensor* t, int32_t dev_id)      { device::dispatchOp(t->device, [&]<Device D>() { ops::PagedAttentionImpl<D>::execute(t, dev_id); }); }
    void flash_attention(Tensor* t, int32_t dev_id){ device::dispatchOp(t->device, [&]<Device D>() { ops::FlashAttentionImpl<D>::execute(t, dev_id); }); }

    // ===== embedding =====
    void embedding(Tensor* t, int32_t dev_id){
        ops::EmbeddingImpl<Device::CPU>::execute(t, dev_id);
    }

    // ===== rope =====
    void apply_rope(Tensor* t, int32_t dev_id)     { device::dispatchOp(t->device, [&]<Device D>() { ops::ApplyRopeImpl<D>::execute(t, dev_id); }); }
    void rope_cache(Tensor* t, int32_t dev_id){
        ops::RopeCacheImpl<Device::CPU>::execute(t, dev_id);
    }

    // ===== shape =====
    void spilt(Tensor* t, int32_t dev_id)         { device::dispatchOp(t->device, [&]<Device D>() { ops::SplitImpl<D>::execute(t, dev_id); }); }

    // cpu -> gpu_x:  dev = gpu, id = x
    // gpu_x -> cpu:  dev = gpu, id = x
    // 不存在直接：gpu0 -> gpu1,已经在外部被拆分为：gpu0 -> cpu,cpu -> gpu1
    void memcpy(Tensor* t, int32_t dev_id) {
        Device dev = t->device;

        Tensor* src = t->src[0];
        Tensor* dst = t;

        if (src && src->device != dst->device) {
            if(dst->device != Device::CPU){
                dev = dst->device;
            }else{
                dev = src->device;
            }
        }
        device::dispatchOp(dev, [&]<Device D>() { ops::MemcpyImpl<D>::execute(t, dev_id); });
    }

    // ===== misc =====
    void concat(Tensor* t, int32_t dev_id)         { device::dispatchOp(t->device, [&]<Device D>() { ops::ConcatImpl<D>::execute(t, dev_id); }); }
    void repeat(Tensor* t, int32_t dev_id)         { device::dispatchOp(t->device, [&]<Device D>() { ops::RepeatImpl<D>::execute(t, dev_id); }); }

    // ===== mamba2 =====
    void causal_conv1d(Tensor* t, int32_t dev_id)  { device::dispatchOp(t->device, [&]<Device D>() { ops::CausalConv1dImpl<D>::execute(t, dev_id); }); }
    void ssm_scan(Tensor* t, int32_t dev_id)       { device::dispatchOp(t->device, [&]<Device D>() { ops::SsmScanImpl<D>::execute(t, dev_id); }); }
} // namespace kernel
