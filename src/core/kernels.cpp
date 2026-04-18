#include "core/kernels.h"
#include "backend/cpu/arithmetic.h"
#include "backend/cpu/normalization.h"
#include "backend/cpu/linear.h"
#include "backend/cpu/shape.h"
#include "backend/cpu/activation.h"
#include "backend/cpu/attention.h"
#include "backend/cpu/embedding.h"
#include "backend/cpu/rope.h"
#include "backend/cpu/sampling.h"
#ifdef BACKEND_CUDA
#include "backend/cuda/arithmetic.cuh"
#include "backend/cuda/normalization.cuh"
#include "backend/cuda/linear.cuh"
#include "backend/cuda/shape.cuh"
#include "backend/cuda/activation.cuh"
#include "backend/cuda/attention.cuh"
#include "backend/cuda/embedding.cuh"
#include "backend/cuda/rope.cuh"
#include "backend/cuda/sampling.cuh"
#endif
#include <format>
#include <stdexcept>

namespace kernel {

    static std::runtime_error not_impl(const char* name, const Tensor* t) {
        return std::runtime_error(std::format("kernel::{} not implemented (tensor: {}, device: {})",
            name, t->name, device_to_string(t->device)));
    }

    // ===== arithmetic =====

    void add(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::add(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::add(t); return; }
#endif
        throw not_impl("add", t);
    }
    void sub(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::sub(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::sub(t); return; }
#endif
        throw not_impl("sub", t);
    }
    void mul(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::mul(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::mul(t); return; }
#endif
        throw not_impl("mul", t);
    }
    void div(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::div(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::div(t); return; }
#endif
        throw not_impl("div", t);
    }
    void scale(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::scale(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::scale(t); return; }
#endif
        throw not_impl("scale", t);
    }

    // ===== normalization =====

    void rms_norm(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::rms_norm(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::rms_norm(t); return; }
#endif
        throw not_impl("rms_norm", t);
    }
    void layer_norm(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::layer_norm(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::layer_norm(t); return; }
#endif
        throw not_impl("layer_norm", t);
    }

    // ===== linear / matmul =====

    void matmul(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::matmul(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::matmul(t); return; }
#endif
        throw not_impl("matmul", t);
    }
    void linear(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::linear(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::linear(t); return; }
#endif
        throw not_impl("linear", t);
    }
    void transpose(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::transpose(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::transpose(t); return; }
#endif
        throw not_impl("transpose", t);
    }

    // ===== shape =====

    void reshape(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::reshape(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::reshape(t); return; }
#endif
        throw not_impl("reshape", t);
    }
    void permute(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::permute(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::permute(t); return; }
#endif
        throw not_impl("permute", t);
    }

    // ===== activation =====

    void silu(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::silu(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::silu(t); return; }
#endif
        throw not_impl("silu", t);
    }
    void gelu(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::gelu(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::gelu(t); return; }
#endif
        throw not_impl("gelu", t);
    }
    void relu(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::relu(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::relu(t); return; }
#endif
        throw not_impl("relu", t);
    }

    // ===== attention =====

    void softmax(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::softmax(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::softmax(t); return; }
#endif
        throw not_impl("softmax", t);
    }
    void diag_mask_inf(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::diag_mask_inf(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::diag_mask_inf(t); return; }
#endif
        throw not_impl("diag_mask_inf", t);
    }
void sdpa(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::sdpa(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::sdpa(t); return; }
#endif
        throw not_impl("sdpa", t);
    }
    void attention(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::attention(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::attention(t); return; }
#endif
        throw not_impl("attention", t);
    }     

    void flash_attention(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::flash_attention(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::flash_attention(t); return; }
#endif
        throw not_impl("flash_attn", t);
    }

    // ===== embedding =====

    void embedding(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::embedding(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::embedding(t); return; }
#endif
        throw not_impl("embedding", t);
    }

    // ===== rope =====

    void apply_rope(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::apply_rope(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::apply_rope(t); return; }
#endif
        throw not_impl("apply_rope", t);
    }
    void rope_cache(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::rope_cache(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::rope_cache(t); return; }
#endif
        throw not_impl("rope_cache", t);
    }

    // ===== misc =====
    void concat(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::concat(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::concat(t); return; }
#endif
        throw not_impl("concat", t);
    }
    void repeat(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::repeat(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::repeat(t); return; }
#endif
        throw not_impl("repeat", t);
    }
    void sampling(Tensor* t) {
        if (t->device == Device::CPU)   { cpu::sampling(t); return; }
#ifdef BACKEND_CUDA
        if (t->device == Device::CUDA)  { cuda::sampling(t); return; }
#endif
        throw not_impl("sampling", t);
    }

} // namespace kernel
