#include "utils/tensor_dump.hpp"
#include "utils/quants.hpp"
#include <fstream>
#include <cstring>

namespace dump {

void dequantize_to_f32(const Tensor* t, float*& out, size_t& num_elems,
                       const void* host_data) {
    num_elems = t->num_elements();
    out = new float[num_elems];

    if (!host_data) host_data = t->data;

    switch (t->dtype) {
    case DataType::GGML_TYPE_F32: {
        std::memcpy(out, host_data, num_elems * sizeof(float));
        break;
    }
    case DataType::GGML_TYPE_BF16: {
        auto* src = static_cast<const bfloat16_t*>(host_data);
        for (size_t i = 0; i < num_elems; ++i)
            out[i] = float(src[i]);
        break;
    }
    case DataType::GGML_TYPE_F16: {
        auto* src = static_cast<const float16_t*>(host_data);
        for (size_t i = 0; i < num_elems; ++i)
            out[i] = float(src[i]);
        break;
    }
    case DataType::GGML_TYPE_Q4_K: {
        auto* w = static_cast<const Block_Q4_K*>(host_data);
        auto* bf16_buf = new bfloat16_t[num_elems];
        quant::dequantize_row_q4_K(w, bf16_buf, (int64_t)num_elems);
        for (size_t i = 0; i < num_elems; ++i)
            out[i] = float(bf16_buf[i]);
        delete[] bf16_buf;
        break;
    }
    case DataType::GGML_TYPE_Q6_K: {
        auto* w = static_cast<const Block_Q6_K*>(host_data);
        auto* bf16_buf = new bfloat16_t[num_elems];
        quant::dequantize_row_q6_K(w, bf16_buf, (int64_t)num_elems);
        for (size_t i = 0; i < num_elems; ++i)
            out[i] = float(bf16_buf[i]);
        delete[] bf16_buf;
        break;
    }
    default: {
        fprintf(stderr, "[tensor_dump] unsupported dtype %d for tensor '%s'\n",
                (int)t->dtype, t->name.c_str());
        delete[] out;
        out = nullptr;
        num_elems = 0;
        break;
    }
    }
}

void save_npy_f32(const std::string& path, const float* data,
                  const int64_t* shape, int ndim) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[tensor_dump] failed to open %s\n", path.c_str());
        return;
    }

    // header
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (int i = 0; i < ndim; ++i) {
        if (i > 0) header += ", ";
        header += std::to_string(shape[i]);
    }
    if (ndim == 1) header += ","; // 1-D must have trailing comma
    header += "), }";

    // align to 64 bytes (numpy v3 style, but works everywhere)
    size_t header_len = header.size() + 1; // +1 for newline
    size_t padding = 64 - (10 + header_len) % 64;
    if (padding == 64) padding = 0;
    header += std::string(padding, ' ') + '\n';
    header_len = header.size();

    // write header
    char magic[8] = {(char)0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    f.write(magic, 8);
    uint16_t hdr_len = (uint16_t)header_len;
    f.write(reinterpret_cast<const char*>(&hdr_len), 2);
    f.write(header.data(), header.size());

    // write data
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) total *= shape[i];
    f.write(reinterpret_cast<const char*>(data), total * sizeof(float));
}

} // namespace tensor_dump
