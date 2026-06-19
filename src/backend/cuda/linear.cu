#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <cublas_v2.h>
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "backend/cuda/linear.h"
#include "utils/dtype_traits.hpp"


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



// ---- GPU 兼容的量化 block 结构体 ----
// 内存布局与 CPU 端 Block_Q4_K / Block_Q6_K 完全一致
// 用 uint16_t 替代 float16_t（二者内存布局相同，均为 IEEE 754 binary16）

struct GpuBlockQ4K {              // 144 bytes
    __half d;                    // super-block scale (fp16 bits)
    __half dmin;                 // super-block min factor (fp16 bits)
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(GpuBlockQ4K) == 144);

struct GpuBlockQ6K {              // 210 bytes
    uint8_t  ql[128];              // lower 4 bits
    uint8_t  qh[64];               // upper 2 bits
    int8_t   scales[16];
    __half d;                    // super-block scale (fp16 bits)
};
static_assert(sizeof(GpuBlockQ6K) == 210);

// ---- helpers ----

static __device__ inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* sc, uint8_t* m){
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)   | ((q[j - 0] >> 6) << 4);
    }
}
// ---- Q4_K 融合 kernel：smem_x 共享，w 直接读 global memory ----
__global__ void linear_q4_k_smem_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const GpuBlockQ4K* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    int M, int K, int N)
{
    extern __shared__ __nv_bfloat16 smem_x[];
    const int out_c = blockIdx.x * blockDim.x + threadIdx.x;
    const int row   = blockIdx.y;
    const bool valid = (out_c < N);

    const int n_super_blocks = K / 256;
    float acc = 0.0f;

    for (int sb = 0; sb < n_super_blocks; ++sb) {
        const __nv_bfloat16* x_sb = x + row * K + sb * 256;
        smem_x[threadIdx.x] = x_sb[threadIdx.x];
        __syncthreads();

        if (valid) {
            const GpuBlockQ4K* w_sb = w + out_c * n_super_blocks + sb;
            const float d    = __half2float(w_sb->d);
            const float dmin = __half2float(w_sb->dmin);
            const uint8_t* scales_ptr = w_sb->scales;
            const uint8_t* q = w_sb->qs;

            for (int j = 0; j < 256; j += 64) {
                uint8_t sc, m;
                int is = j / 32;
                get_scale_min_k4(is + 0, scales_ptr, &sc, &m);
                const float d1 = d * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, scales_ptr, &sc, &m);
                const float d2 = d * sc, m2 = dmin * m;

                for (int l = 0; l < 32; ++l) {
                    float w_lo = d1 * (q[l] & 0xF) - m1;
                    float w_hi = d2 * (q[l] >> 4) - m2;
                    acc += __bfloat162float(smem_x[j + l]) * w_lo;
                    acc += __bfloat162float(smem_x[j + 32 + l]) * w_hi;
                }
                q += 32;
            }
        }
        __syncthreads();
    }

    if (valid) {
        if (bias != nullptr)
            acc += __bfloat162float(bias[out_c]);
        out[row * N + out_c] = __float2bfloat16(acc);
    }
}

// ---- Q6_K 融合 kernel：smem_x 共享，w 直接读 global memory ----
__global__ void linear_q6_k_smem_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    const GpuBlockQ6K* __restrict__ w,
    const __nv_bfloat16* __restrict__ bias,
    int M, int K, int N)
{
    extern __shared__ __nv_bfloat16 smem_x[];
    const int out_c = blockIdx.x * blockDim.x + threadIdx.x;
    const int row   = blockIdx.y;
    const bool valid = (out_c < N);

    const int n_super_blocks = K / 256;
    float acc = 0.0f;

    for (int sb = 0; sb < n_super_blocks; ++sb) {
        const __nv_bfloat16* x_sb = x + row * K + sb * 256;
        smem_x[threadIdx.x] = x_sb[threadIdx.x];
        __syncthreads();

        if (valid) {
            const GpuBlockQ6K* w_sb = w + out_c * n_super_blocks + sb;
            const float d = __half2float(w_sb->d);
            const uint8_t* ql = w_sb->ql;
            const uint8_t* qh = w_sb->qh;
            const int8_t*  sc = w_sb->scales;

            for (int n = 0; n < 256; n += 128) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                    acc += __bfloat162float(smem_x[n + l +  0]) * (d * sc[is + 0] * q1);
                    acc += __bfloat162float(smem_x[n + l + 32]) * (d * sc[is + 2] * q2);
                    acc += __bfloat162float(smem_x[n + l + 64]) * (d * sc[is + 4] * q3);
                    acc += __bfloat162float(smem_x[n + l + 96]) * (d * sc[is + 6] * q4);
                }
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        __syncthreads();
    }

    if (valid) {
        if (bias != nullptr)
            acc += __bfloat162float(bias[out_c]);
        out[row * N + out_c] = __float2bfloat16(acc);
    }
}

void LinearImpl<Device::CUDA>::execute(Tensor* out, int32_t dev_id){
    const Tensor* x    = out->src[0]; // [B, M, K]
    const Tensor* w    = out->src[1]; // [N, K]
    const Tensor* bias = out->src[2]; // 似乎不用

    int64_t M    = x->dims[1]; 
    int64_t K    = x->dims[2]; 
    int64_t N    = w->dims[0];

    constexpr int threads = 256;
    const int shared_mem = 256 * sizeof(__nv_bfloat16);  // smem_x only, 512 bytes
    auto* bias_ptr = (bias && bias->data)
        ? static_cast<const __nv_bfloat16*>(bias->data) : nullptr;

    switch(w->dtype){
        case DataType::GGML_TYPE_Q4_K:{
            const int blocks_x = (static_cast<int>(N) + threads - 1) / threads;
            linear_q4_k_smem_kernel<<<dim3(blocks_x, static_cast<int>(M)), threads, shared_mem>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(x->data),
                static_cast<const GpuBlockQ4K*>(w->data),
                bias_ptr,
                M, K, N
            );
            return;
        }
        case DataType::GGML_TYPE_Q6_K:{
            const int blocks_x = (static_cast<int>(N) + threads - 1) / threads;
            linear_q6_k_smem_kernel<<<dim3(blocks_x, static_cast<int>(M)), threads, shared_mem>>>(
                static_cast<__nv_bfloat16*>(out->data),
                static_cast<const __nv_bfloat16*>(x->data),
                static_cast<const GpuBlockQ6K*>(w->data),
                bias_ptr,
                M, K, N
            );
            return;
        }
    }


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
    const int ldb = K, ldc = N;
    const long long strideA = 0;
    const long long strideB = M * K;
    const long long strideC = M * N;

    cublasHandle_t handle = CublasHandlePool::get();
    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        A, dtype, K, strideA,
        Bptr, dtype, ldb, strideB,
        &beta,
        C, dtype, ldc, strideC,
        1,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (bias && bias->data) {
        launch_add_bias(out, bias, 1, static_cast<int>(M), static_cast<int>(N));
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Linear GEMM launch failed: %s\n", cudaGetErrorString(err));
    }
}

void MatmulImpl<Device::CUDA>::execute(Tensor*, int32_t)    { throw std::runtime_error("cuda::matmul not implemented"); }

template struct LinearImpl<Device::CUDA>;
template struct MatmulImpl<Device::CUDA>;
}
