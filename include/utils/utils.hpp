#pragma once
#include <cstdint>
#include <string>

// 支持的后端设备
enum class Device: uint8_t {
    CPU = 0,
    CUDA = 1,
    SYCL = 3,
    VULKAN = 4,
    AUTO = 255
};
enum class TensorType : uint8_t {
    TENSOR_TYPE_UNKNOWN = 0,
    // [静态]   只需计算/加载一次，整个推理过程中保持不变
    TENSOR_TYPE_WEIGHT,                   // 模型权重 (如 attn_q.weight, ffn_down.weight)
    TENSOR_TYPE_CACHE,                    // 预计算缓存 (如 RoPE cos/sin 表)
    TENSOR_TYPE_KV_CACHE,                 // KV Cache (跨 step 复用，特殊管理)
    // [动态] 每轮推理都需要计算/更新
    TENSOR_TYPE_INPUT,                    // 用户输入 (如 input_ids, position_ids)
    TENSOR_TYPE_ACTIVATION,               // 中间激活值 (如 q_proj, attn_out, ffn_inter)
    TENSOR_TYPE_OUTPUT,                   // 最终输出 (如 logits)
    // [特殊] 视图类型，不占用实际内存，多个视图共享同一数据块
    TENSOR_TYPE_VIEW,                     // 视图 (reshape/permute 创建，共享内存)
};
// 权重数据类型枚举
enum class DataType:uint8_t {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    GGML_TYPE_MXFP4   = 39,
    GGML_TYPE_COUNT   = 40
};
enum class GGUFType : uint8_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
};
enum class OperationType:uint8_t {
    OP_TYPE_NONE         ,
    OP_TYPE_MEMCPY       ,
    OP_TYPE_DUP          ,
    OP_TYPE_ADD          ,
    OP_TYPE_SUB          ,
    OP_TYPE_MUL          ,
    OP_TYPE_DIV          ,
    OP_TYPE_SCALE        ,
    OP_TYPE_MAT_MUL      ,
    OP_TYPE_TRANSPOSE    ,
    OP_TYPE_RESHAPE      ,
    OP_TYPE_PERMUTE      ,
    OP_TYPE_VIEW         ,
    OP_TYPE_CONCAT       ,
    OP_TYPE_REPEAT       ,
    OP_TYPE_SOFTMAX      ,
    OP_TYPE_RMS_NORM     ,
    OP_TYPE_LAYER_NORM   ,
    OP_TYPE_GELU         ,
    OP_TYPE_SILU         ,
    OP_TYPE_RELU         ,
    OP_TYPE_DIAG_MASK_INF,
    OP_TYPE_POOL_2D      ,
    OP_TYPE_UPSCALE      ,
    OP_TYPE_PAD          ,
    OP_TYPE_UNPAD        ,
    OP_TYPE_PLACEHOLDER  ,
    OP_TYPE_EMBEDDING    ,
    OP_TYPE_LINEAR       ,
    OP_TYPE_APPLY_ROPE,
    OP_TYPE_SDPA,
    OP_TYPE_TOKENIZE,
    OP_TYPE_SAMPLING,
    OP_TYPE_ROPE_CACHE,
    OP_TYPE_CONV2D,
    OP_TYPE_FLASH_ATTN
};
// 模型类型
enum class ModelType:uint8_t {
    UNKNOWN,
    CAUSAL_LM,      // 因果语言模型
    EMBEDDING,      // 嵌入模型
    SEQ2SEQ,        // 序列到序列
    CLASSIFIER,     // 分类器
};

// 模型架构类型
enum class ModelArch:uint8_t{
    UNKNOWN,
    QWEN3
};


// 辅助函数：GGUF 类型转字符串
inline std::string gguf_type_to_string(GGUFType type) {
    switch (type) {
        case GGUFType::GGUF_TYPE_UINT8:   return "UINT8";
        case GGUFType::GGUF_TYPE_INT8:    return "INT8";
        case GGUFType::GGUF_TYPE_UINT16:  return "UINT16";
        case GGUFType::GGUF_TYPE_INT16:   return "INT16";
        case GGUFType::GGUF_TYPE_UINT32:  return "UINT32";
        case GGUFType::GGUF_TYPE_INT32:   return "INT32";
        case GGUFType::GGUF_TYPE_FLOAT32: return "FLOAT32";
        case GGUFType::GGUF_TYPE_BOOL:    return "BOOL";
        case GGUFType::GGUF_TYPE_STRING:  return "STRING";
        case GGUFType::GGUF_TYPE_ARRAY:   return "ARRAY";
        case GGUFType::GGUF_TYPE_UINT64:  return "UINT64";
        case GGUFType::GGUF_TYPE_INT64:   return "INT64";
        case GGUFType::GGUF_TYPE_FLOAT64: return "FLOAT64";
        default:                          return "UNKNOWN";
    }
}

// 辅助函数：获取数据类型的字节大小(byte)
inline size_t data_type_size(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::GGML_TYPE_F32:     return 4;
        case DataType::GGML_TYPE_F16:     return 2;
        case DataType::GGML_TYPE_Q4_0:    return sizeof(float) + 4;    // super-block + scale
        case DataType::GGML_TYPE_Q4_1:    return 2 * sizeof(float) + 4; // super-block + scales
        case DataType::GGML_TYPE_Q5_0:    return sizeof(float) + 6;     // super-block + scale
        case DataType::GGML_TYPE_Q5_1:    return 2 * sizeof(float) + 6; // super-block + scales
        case DataType::GGML_TYPE_Q8_0:    return sizeof(float) + 8;     // super-block + scale
        case DataType::GGML_TYPE_Q8_1:    return 2 * sizeof(float) + 8; // super-block + scales
        case DataType::GGML_TYPE_Q2_K:    return 256;  // Q2_K block size
        case DataType::GGML_TYPE_Q3_K:    return 256;  // Q3_K block size
        case DataType::GGML_TYPE_Q4_K:    return 256;  // Q4_K block size
        case DataType::GGML_TYPE_Q5_K:    return 256;  // Q5_K block size
        case DataType::GGML_TYPE_Q6_K:    return 256;  // Q6_K block size
        case DataType::GGML_TYPE_Q8_K:    return 256;  // Q8_K block size
        case DataType::GGML_TYPE_IQ2_XXS: return 256;  // IQ2_XXS block size
        case DataType::GGML_TYPE_IQ2_XS:  return 256;  // IQ2_XS block size
        case DataType::GGML_TYPE_IQ3_XXS: return 256;  // IQ3_XXS block size
        case DataType::GGML_TYPE_IQ1_S:   return 256;  // IQ1_S block size
        case DataType::GGML_TYPE_IQ4_NL:  return 256;  // IQ4_NL block size
        case DataType::GGML_TYPE_IQ3_S:   return 256;  // IQ3_S block size
        case DataType::GGML_TYPE_IQ2_S:   return 256;  // IQ2_S block size
        case DataType::GGML_TYPE_IQ4_XS:  return 256;  // IQ4_XS block size
        case DataType::GGML_TYPE_I8:      return 1;
        case DataType::GGML_TYPE_I16:     return 2;
        case DataType::GGML_TYPE_I32:     return 4;
        case DataType::GGML_TYPE_I64:     return 8;
        case DataType::GGML_TYPE_F64:     return 8;
        case DataType::GGML_TYPE_IQ1_M:   return 256;  // IQ1_M block size
        case DataType::GGML_TYPE_BF16:    return 2;
        case DataType::GGML_TYPE_TQ1_0:   return 256;  // TQ1_0 block size
        case DataType::GGML_TYPE_TQ2_0:   return 256;  // TQ2_0 block size
        case DataType::GGML_TYPE_MXFP4:   return 256;  // MXFP4 block size
        case DataType::GGML_TYPE_COUNT:   return 0;
        default:                          return 0;
    }
}
// 辅助函数：操作类型转字符串
inline std::string operation_type_to_string(OperationType op) {
    switch(op){
        case OperationType::OP_TYPE_NONE:          return "None";
        case OperationType::OP_TYPE_DUP:           return "Dup";
        case OperationType::OP_TYPE_ADD:           return "Add";
        case OperationType::OP_TYPE_SUB:           return "Sub";
        case OperationType::OP_TYPE_MUL:           return "Mul";
        case OperationType::OP_TYPE_DIV:           return "Div";
        case OperationType::OP_TYPE_SCALE:         return "Scale";
        case OperationType::OP_TYPE_MAT_MUL:       return "MatMul";
        case OperationType::OP_TYPE_TRANSPOSE:     return "Transpose";
        case OperationType::OP_TYPE_RESHAPE:       return "Reshape";
        case OperationType::OP_TYPE_PERMUTE:       return "Permute";
        case OperationType::OP_TYPE_VIEW:          return "View";
        case OperationType::OP_TYPE_CONCAT:        return "Concat";
        case OperationType::OP_TYPE_REPEAT:        return "Repeat";
        case OperationType::OP_TYPE_SOFTMAX:       return "Softmax";
        case OperationType::OP_TYPE_RMS_NORM:      return "RMSNorm";
        case OperationType::OP_TYPE_LAYER_NORM:    return "LayerNorm";
        case OperationType::OP_TYPE_GELU:          return "GELU";
        case OperationType::OP_TYPE_SILU:          return "SiLU";
        case OperationType::OP_TYPE_RELU:          return "ReLU";
        case OperationType::OP_TYPE_DIAG_MASK_INF: return "DiagMaskInf";
        case OperationType::OP_TYPE_POOL_2D:       return "Pool2D";
        case OperationType::OP_TYPE_UPSCALE:       return "Upscale";
        case OperationType::OP_TYPE_PAD:           return "Pad";
        case OperationType::OP_TYPE_UNPAD:         return "Unpad";
        case OperationType::OP_TYPE_MEMCPY:        return "Memcpy";
        case OperationType::OP_TYPE_PLACEHOLDER:   return "Placeholder";
        case OperationType::OP_TYPE_EMBEDDING:     return "Embedding";
        case OperationType::OP_TYPE_LINEAR:        return "Linear";
        case OperationType::OP_TYPE_APPLY_ROPE:    return "Rope";
        case OperationType::OP_TYPE_SDPA:          return "SDPA";
        case OperationType::OP_TYPE_TOKENIZE:      return "Tokenize";
        case OperationType::OP_TYPE_SAMPLING:      return "Sampling";
        case OperationType::OP_TYPE_ROPE_CACHE:    return "RopeCache";
        case OperationType::OP_TYPE_CONV2D:        return "Conv2D";
        case OperationType::OP_TYPE_FLASH_ATTN:    return "FlashAttn";
        default:                                   return "Unknown";
    }
}

// 辅助函数：数据类型转字符串
inline std::string data_type_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::GGML_TYPE_F32:     return "F32";
        case DataType::GGML_TYPE_F16:     return "F16";
        case DataType::GGML_TYPE_Q4_0:    return "Q4_0";
        case DataType::GGML_TYPE_Q4_1:    return "Q4_1";
        case DataType::GGML_TYPE_Q5_0:    return "Q5_0";
        case DataType::GGML_TYPE_Q5_1:    return "Q5_1";
        case DataType::GGML_TYPE_Q8_0:    return "Q8_0";
        case DataType::GGML_TYPE_Q8_1:    return "Q8_1";
        case DataType::GGML_TYPE_Q2_K:    return "Q2_K";
        case DataType::GGML_TYPE_Q3_K:    return "Q3_K";
        case DataType::GGML_TYPE_Q4_K:    return "Q4_K";
        case DataType::GGML_TYPE_Q5_K:    return "Q5_K";
        case DataType::GGML_TYPE_Q6_K:    return "Q6_K";
        case DataType::GGML_TYPE_Q8_K:    return "Q8_K";
        case DataType::GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case DataType::GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case DataType::GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case DataType::GGML_TYPE_IQ1_S:   return "IQ1_S";
        case DataType::GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case DataType::GGML_TYPE_IQ3_S:   return "IQ3_S";
        case DataType::GGML_TYPE_IQ2_S:   return "IQ2_S";
        case DataType::GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        case DataType::GGML_TYPE_I8:      return "I8";
        case DataType::GGML_TYPE_I16:     return "I16";
        case DataType::GGML_TYPE_I32:     return "I32";
        case DataType::GGML_TYPE_I64:     return "I64";
        case DataType::GGML_TYPE_F64:     return "F64";
        case DataType::GGML_TYPE_IQ1_M:   return "IQ1_M";
        case DataType::GGML_TYPE_BF16:    return "BF16";
        case DataType::GGML_TYPE_TQ1_0:   return "TQ1_0";
        case DataType::GGML_TYPE_TQ2_0:   return "TQ2_0";
        case DataType::GGML_TYPE_MXFP4:   return "MXFP4";
        case DataType::GGML_TYPE_COUNT:   return "COUNT";
        default:                          return "Unknown";
    }
}
// 辅助函数：模型类型转字符串
inline std::string model_arch_to_string(ModelArch arch) {
    switch (arch) {
        case ModelArch::QWEN3:      return "Qwen3";
        default:                   return "Unknown";
    }
}
inline std::string model_type_to_string(ModelType type) {
    switch (type) {
        case ModelType::CAUSAL_LM:  return "CausalLM";
        case ModelType::EMBEDDING:  return "Embedding";
        case ModelType::SEQ2SEQ:    return "Seq2Seq";
        case ModelType::CLASSIFIER: return "Classifier";
        default:                    return "Unknown";
    }
}
inline std::string device_to_string(Device dev) {
    switch (dev) {
        case Device::CPU: return "CPU";
        case Device::CUDA: return "CUDA";
        case Device::SYCL: return "SYCL";
        case Device::VULKAN: return "VULKAN";
        case Device::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}