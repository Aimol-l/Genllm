#include <format>
#include <stdexcept>
#include "backend/cuda/linear.cuh"
#include <cuda_runtime.h>

namespace cuda {

void matmul(Tensor* out) {
    // TODO: matrix multiplication (CUDA kernel or cuBLAS)
    // src[0]: [M, K], src[1]: [K, N] -> out: [M, N]
    // dtype dispatch:
    //   F32: cublasSgemm / cutlass
    //   F16: cublasHgemm (Tensor Core)
    //   BF16: cublasGemmEx with CUBLAS_COMPUTE_32F (Ampere+)
    //   Q4/Q8: dequantize kernel -> F16/BF16 -> cublasHgemm
    // tip: consider using cuBLASLt for flexible layout & compute type
    throw std::runtime_error(std::format("cuda::matmul not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void linear(Tensor* out) {
    // TODO: fully-connected layer (CUDA kernel or cuBLAS)
    // src[0]: input [batch, in_features], src[1]: weight, src[2]: bias
    // out = input @ weight^T + bias
    // tip: cublasGemmEx for matmul, then element-wise add for bias
    throw std::runtime_error(std::format("cuda::linear not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

void transpose(Tensor* out) {
    // TODO: transpose (CUDA kernel or cuBLAS geam)
    // tip: cublasSgeam/Cgeam with alpha=1, beta=0, opA=CUBLAS_OP_T
    //       or custom kernel for better perf on non-contiguous tensors
    throw std::runtime_error(std::format("cuda::transpose not implemented (dtype: {})",
        data_type_to_string(out->dtype)));
}

} // namespace cuda
