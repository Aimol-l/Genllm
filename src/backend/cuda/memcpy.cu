#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
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

        if (src_dev == Device::CPU && dst_dev == Device::CUDA) {
            cudaMemcpy(out->data, src->data, nbytes, cudaMemcpyHostToDevice);
            return;
        }

        if (src_dev == Device::CUDA && dst_dev == Device::CPU) {
            cudaMemcpy(out->data, src->data, nbytes, cudaMemcpyDeviceToHost);
            return;
        }

        if (src_dev == Device::CUDA && dst_dev == Device::CUDA) {
            cudaPointerAttributes src_attr, dst_attr;
            cudaPointerGetAttributes(&src_attr, src->data);
            cudaPointerGetAttributes(&dst_attr, out->data);

            if (src_attr.device == dst_attr.device) {
                cudaMemcpy(out->data, src->data, nbytes, cudaMemcpyDeviceToDevice);
            } else {
                // 跨设备：CUDA:0 → host → CUDA:1
                int dst_id = dst_attr.device;
                std::vector<std::byte> staging(nbytes);
                cudaSetDevice(src_attr.device);
                cudaMemcpy(staging.data(), src->data, nbytes, cudaMemcpyDeviceToHost);
                cudaSetDevice(dst_id);
                cudaMemcpy(out->data, staging.data(), nbytes, cudaMemcpyHostToDevice);
            }
            return;
        }

        throw std::runtime_error(
            "MemcpyImpl<CUDA>: unsupported copy (" +
            device_to_string(src_dev) + " -> " + device_to_string(dst_dev) + ")");
    }

    template struct MemcpyImpl<Device::CUDA>;
}
