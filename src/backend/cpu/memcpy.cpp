#include <cstring>
#include <stdexcept>
#include "backend/cpu/memcpy.h"
#include "utils/utils.hpp"

namespace ops {

    void MemcpyImpl<Device::CPU>::execute(Tensor* out) {
        Tensor* src = out->src[0];
        if (!src || !src->data) {
            throw std::runtime_error("MemcpyImpl<CPU>: source tensor has no data");
        }
        size_t nbytes = out->bytes();
        Device src_dev = src->device;
        Device dst_dev = out->device;

        if (src_dev == dst_dev) {
            std::memcpy(out->data, src->data, nbytes);
            return;
        }

        throw std::runtime_error(
            "MemcpyImpl<CPU>: cross-device copy requires CUDA backend (" +
            device_to_string(src_dev) + " -> " + device_to_string(dst_dev) + ")");
    }

    template struct MemcpyImpl<Device::CPU>;
}

