#include <cuda_runtime.h>
#include <stdexcept>
#include "backend/cuda/memcpy.h"
#include "utils/utils.hpp"

namespace ops {

    void MemcpyImpl<Device::CUDA>::execute(Tensor* out) {
        Tensor* src = out->src[0];
        if (!src || !src->data) {
            throw std::runtime_error("MemcpyImpl<CUDA>: source tensor has no data");
        }
        size_t nbytes = out->bytes();
        Device src_dev = src->device;
        Device dst_dev = out->device;

        if (src_dev == dst_dev && src_dev == Device::CUDA) {
            cudaMemcpy(out->data, src->data, nbytes, cudaMemcpyDeviceToDevice);
            return;
        }

        if (src_dev == Device::CPU && dst_dev == Device::CUDA) {
            cudaMemcpy(out->data, src->data, nbytes, cudaMemcpyHostToDevice);
            return;
        }

        if (src_dev == Device::CUDA && dst_dev == Device::CPU) {
            cudaMemcpy(out->data, src->data, nbytes, cudaMemcpyDeviceToHost);
            return;
        }

        throw std::runtime_error(
            "MemcpyImpl<CUDA>: unsupported copy (" +
            device_to_string(src_dev) + " -> " + device_to_string(dst_dev) + ")");
    }

    template struct MemcpyImpl<Device::CUDA>;
}

