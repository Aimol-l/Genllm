#include <stdexcept>
#include "utils/dtype_traits.hpp"
#include "backend/cuda/linear.h"
#include <cublas_v2.h>
#include "cuda_fp16.h"
#include "cuda_bf16.h"

namespace ops {

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
            add_bias_kernel<T><<<blocks, threads>>>(
                out_ptr,
                bias_ptr,
                B, M, N
            );
        }else if constexpr(std::is_same_v<T,__nv_bfloat16>){
            __nv_bfloat16 * out_ptr = static_cast<__nv_bfloat16*>(out->data);
            const __nv_bfloat16* bias_ptr = static_cast<const __nv_bfloat16*>(bias->data);
            add_bias_kernel<T><<<blocks, threads>>>(
                out_ptr,
                bias_ptr,
                B, M, N
            );
        }
    });
}
void LinearImpl<Device::CUDA>::execute(Tensor* out){
    const Tensor* x    = out->src[0]; // [B, M, K]
    const Tensor* w    = out->src[1]; // [K, N] or [N, K]
    const Tensor* bias = out->src[2];
    bool transpose_w   = out->op_params[0] == 1;
    const int64_t B = x->dims[0];
    const int64_t M = x->dims[1];
    const int64_t K = x->dims[2];
    const int64_t N = transpose_w ? w->dims[0] : w->dims[1];
    // ===== dtype & compute =====
    cudaDataType_t dtype;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if (x->dtype == DataType::GGML_TYPE_BF16) {
        dtype = CUDA_R_16BF;
    } else if (x->dtype == DataType::GGML_TYPE_F16) {
        dtype = CUDA_R_16F;
    } else {
        throw std::runtime_error("Linear only supports fp16/bf16");
    }
    // ===== handle =====
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const void* A = w->data;      // 权重不分 batch
    const void* Bptr = x->data;   // 输入 batched
    void*       C = out->data;
    cublasOperation_t opA = transpose_w ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_N;
    const int lda = transpose_w ? K : N; // W
    const int ldb = K;                  // X
    const int ldc = N;                  // OUT
    const long long strideA = 0;             // W 共享
    const long long strideB = M * K;         // 每个 batch 的 X
    const long long strideC = M * N;         // 每个 batch 的 OUT
    cublasGemmStridedBatchedEx(
        handle,
        opA, opB,
        /*m*/ N,
        /*n*/ M,
        /*k*/ K,
        &alpha,
        A, dtype, lda, strideA,
        Bptr, dtype, ldb, strideB,
        &beta,
        C, dtype, ldc, strideC,
        /*batchCount*/ B,
        compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (bias) {
        launch_add_bias(out, bias, B, M, N);
    }
    cublasDestroy(handle);
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
