#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include "utils/dtype_traits.hpp"
#include "backend/cuda/linear.h"
#include <cublas_v2.h>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {
namespace {

    class CublasHandlePool {
    public:
        static cublasHandle_t get() {
            int dev;
            cudaGetDevice(&dev);
            return instance().get_impl(dev);
        }
    private:
        static CublasHandlePool& instance() {
            static CublasHandlePool pool;
            return pool;
        }
        cublasHandle_t get_impl(int dev) {
            std::lock_guard lock(mutex_);
            auto it = handles_.find(dev);
            if (it != handles_.end()) return it->second;
            cublasHandle_t h;
            cublasCreate(&h);
            handles_[dev] = h;
            return h;
        }
        ~CublasHandlePool() {
            for (auto& [_, h] : handles_) cublasDestroy(h);
        }
        std::unordered_map<int, cublasHandle_t> handles_;
        std::mutex mutex_;
    };

    template<typename T>
    __global__ void add_bias_kernel(
        T* out,
        const T* bias,
        int B, int M, int N
    ){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * M * N;
        if (idx >= total) return;
        int n = idx % N;
        out[idx] = out[idx] + bias[n];
    }

    void launch_add_bias(Tensor* out, const Tensor* bias, int B, int M, int N){
        int total = B * M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        dtype::dispatch(out->dtype, [&]<DataType D>() {
            using T = dtype::type_t<D>;
            if constexpr (std::is_same_v<T,__half>) {
                __half * out_ptr = static_cast<__half*>(out->data);
                const __half* bias_ptr = static_cast<const __half*>(bias->data);
                add_bias_kernel<T><<<blocks, threads>>>(out_ptr, bias_ptr, B, M, N);
            } else if constexpr(std::is_same_v<T,__nv_bfloat16>){
                __nv_bfloat16 * out_ptr = static_cast<__nv_bfloat16*>(out->data);
                const __nv_bfloat16* bias_ptr = static_cast<const __nv_bfloat16*>(bias->data);
                add_bias_kernel<T><<<blocks, threads>>>(out_ptr, bias_ptr, B, M, N);
            }
        });
    }
} // anonymous namespace

void LinearImpl<Device::CUDA>::execute(Tensor* out){
    const Tensor* x    = out->src[0]; // [B, M, K]
    const Tensor* w    = out->src[1]; // [K, N] or [N, K]
    const Tensor* bias = out->src[2];
    bool transpose_w   = out->op_params[0] == 1;
    const int64_t B = x->dims[0];
    const int64_t M = x->dims[1];
    const int64_t K = x->dims[2];
    const int64_t N = transpose_w ? w->dims[0] : w->dims[1];

    cudaDataType_t dtype;
    if (x->dtype == DataType::GGML_TYPE_BF16) {
        dtype = CUDA_R_16BF;
    } else if (x->dtype == DataType::GGML_TYPE_F16) {
        dtype = CUDA_R_16F;
    } else {
        throw std::runtime_error("Linear only supports fp16/bf16");
    }

    const float alpha = 1.0f, beta = 0.0f;
    const void* A = w->data;
    const void* Bptr = x->data;
    void*       C = out->data;
    cublasOperation_t opA = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda = transpose_w ? K : N;
    const int ldb = K, ldc = N;
    const long long strideA = 0;
    const long long strideB = M * K;
    const long long strideC = M * N;

    cublasHandle_t handle = CublasHandlePool::get();
    cublasGemmStridedBatchedEx(
        handle,
        opA, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        A, dtype, lda, strideA,
        Bptr, dtype, ldb, strideB,
        &beta,
        C, dtype, ldc, strideC,
        B,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (bias && bias->data) {
        launch_add_bias(out, bias, static_cast<int>(B), static_cast<int>(M), static_cast<int>(N));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Linear GEMM launch failed: %s\n", cudaGetErrorString(err));
    }
}

void MatmulImpl<Device::CUDA>::execute(Tensor*)    { throw std::runtime_error("cuda::matmul not implemented"); }
void TransposeImpl<Device::CUDA>::execute(Tensor*) { throw std::runtime_error("cuda::transpose not implemented"); }

template struct LinearImpl<Device::CUDA>;
template struct MatmulImpl<Device::CUDA>;
template struct TransposeImpl<Device::CUDA>;
}
