#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include "utils/tools.hpp"
#include "utils/utils.hpp"
#include "core/tensor.hpp"
#include "core/gguf_parser.h"


// ============================================================================
// OpFactory: 通用算子节点创建 (只建图，不分配内存)
// 所有函数返回新 Tensor*，用户负责生命周期管理
// ============================================================================
struct OpFactory {
    // 分配到权重区，但不是权重
    static Tensor* placeholder(
        DataType dtype,
        TensorType type, 
        std::initializer_list<int64_t> dims, 
        const std::string& name,
        int32_t layer_id = -1
    ){
        if(dims.size()>4){
            LOG_ERROR("dims size must 1~4");
            throw std::runtime_error("dims size must 1~4");
        }
        Tensor* t = new Tensor();
        t->name = name;
        t->dtype = dtype;
        t->type = type;
        t->layer_id = layer_id;
        t->op_type = OperationType::OP_TYPE_NONE;
        std::copy(dims.begin(), dims.end(), t->dims.begin());
        OpFactory::compute_strides(t);
        t->data = nullptr;  // 等待外部绑定
        return t;
    }
    // 静态权重占位符 (从 GGUF 加载，data=nullptr，执行前绑定)
    static Tensor* weight_placeholder(const TensorInfo* info, const std::string& name, int32_t layer_id = -1){
        Tensor* t = new Tensor();
        t->name = name;
        t->layer_id = layer_id;
        t->dtype = info->dtype;
        t->type = TensorType::TENSOR_TYPE_WEIGHT;
        t->op_type = OperationType::OP_TYPE_NONE;
        t->offset = info->offset;
        std::copy(info->dimensions.begin(), info->dimensions.end(), t->dims.begin());
        OpFactory::compute_strides(t);
        t->data = nullptr;
        return t;
    }
    // ──────────────────────────────────────────────────────────────────
    // linear: y = x @ weight + bias
    // ──────────────────────────────────────────────────────────────────
    static Tensor* linear(
        Tensor* input, 
        const TensorInfo* weight_info,
        const std::string& name = "",
        int32_t layer_id = -1,
        Tensor* bias = nullptr
    ){
        Tensor* t = new Tensor();
        t->name = name;
        t->layer_id = layer_id;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->dtype = input->dtype;
        t->op_type = OperationType::OP_TYPE_LINEAR;

        auto out_dims = infer_linear_output(
            {input->dims.begin(), input->dims.end()},  // 传整个数组
            weight_info->dimensions,
            true 
        );
        std::copy(out_dims.begin(), out_dims.end(), t->dims.begin());

        t->src[0] = input;
        t->src[1] = OpFactory::weight_placeholder(weight_info, weight_info->name,layer_id);
        t->src[2] = bias;
        OpFactory::compute_strides(t);
        return t;
    }

    static Tensor* embedding_lookup(
        Tensor* input_ids,
        const TensorInfo* weight_info,
        const std::string& name = "",
        int32_t layer_id = -1,
        bool transpose = false,
        DataType activation_dtype = DataType::GGML_TYPE_BF16
    ){
        Tensor* t = new Tensor();
        t->name  = name;
        t->dtype = is_quantized(weight_info->dtype) ? activation_dtype : weight_info->dtype;
        t->type  = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_EMBEDDING;

        t->dims[0] = input_ids->dims[0];  // batch
        t->dims[1] = input_ids->dims[1];  // seq_len

        t->dims[2] = transpose ? weight_info->dimensions[0]:weight_info->dimensions[1];  // hidden_size

        t->src[0]  = input_ids;
        t->src[1]  = OpFactory::weight_placeholder(weight_info, weight_info->name,layer_id);
        t->op_params[0] = transpose ? 1 : 0;  // 转置标志
        OpFactory::compute_strides(t);

        return t;
    }
    // ──────────────────────────────────────────────────────────────────
    // rms_norm: y = (x / sqrt(mean(x²)+eps)) * weight
    // ──────────────────────────────────────────────────────────────────
    static Tensor* rms_norm(
        Tensor* input,      // [batch, seq_len, hidden_size] or  [batch, num_heads, seq_len, head_dim]
        const TensorInfo* weight_info,
        float eps,
        const std::string& name = "",
        int32_t layer_id = -1,
        float w_offset = 0.0f
    ){
        Tensor* t = new Tensor();
        t->name = name;
        t->layer_id = layer_id;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->dtype = input->dtype;
        t->op_type = OperationType::OP_TYPE_RMS_NORM;
        // 输出形状同输入
        std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());
        t->src[0] = input;
        t->src[1] = OpFactory::weight_placeholder(weight_info, weight_info->name,layer_id);
        t->op_params[0] = eps;
        t->op_params[1] = w_offset;
        OpFactory::compute_strides(t);
        return t;
    }
    static std::tuple<Tensor*, Tensor*> rope_cache(
        int max_seq_len,
        int rope_dim, 
        float theta, 
        DataType dtype, 
        const std::string& name_prefix = "rope",
        int32_t layer_id = -1
    ){
        auto make_cache_tensor = [&](const std::string& name) -> Tensor* {
            Tensor* t = new Tensor();
            t->name = name;
            t->dtype = dtype;
            t->type = TensorType::TENSOR_TYPE_CACHE;
            t->op_type = OperationType::OP_TYPE_ROPE_CACHE;
            // 形状: [max_seq_len, rope_dim]
            t->dims[0] = max_seq_len;
            t->dims[1] = rope_dim;
            for (int i = 2; i < TENSOR_MAX_DIMS; ++i) t->dims[i] = 0;
            // 步长: 行优先 (元素跨度)
            t->strides[1] = 1;
            t->strides[0] = rope_dim;
            for (int i = 2; i < TENSOR_MAX_DIMS; ++i) t->strides[i] = 0;
            t->data = nullptr;
            t->offset = 0;
            t->layer_id = layer_id;
            return t;
        };
        Tensor* cos_tensor = make_cache_tensor(name_prefix + "_cos");
        Tensor* sin_tensor = make_cache_tensor(name_prefix + "_sin");
        sin_tensor->op_params[0] = theta;
        sin_tensor->op_params[1] = rope_dim;
        sin_tensor->op_params[2] = max_seq_len;
        cos_tensor->op_params[0] = theta;
        cos_tensor->op_params[1] = rope_dim;
        cos_tensor->op_params[2] = max_seq_len;
        return {cos_tensor, sin_tensor};
    }
    // y = x * sigmoid(x)
    static Tensor* silu(Tensor* input, const std::string& name = "", int32_t layer_id = -1){
        Tensor* t = new Tensor;
        t->name = name;
        t->layer_id = layer_id;
        t->dtype = input->dtype;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;

        t->op_type = OperationType::OP_TYPE_SILU;
        std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());
        t->src[0] = input;
        OpFactory::compute_strides(t);
        return t;
    }
    // y = sigmoid(x)
    static Tensor* sigmoid(Tensor* input, const std::string& name = "", int32_t layer_id = -1){
        Tensor* t = new Tensor;
        t->name = name;
        t->layer_id = layer_id;
        t->dtype = input->dtype;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_SIGMOID;
        std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());
        t->src[0] = input;
        OpFactory::compute_strides(t);
        return t;
    }
    // split: 沿指定维度切片, op_params[0]=dim, [1]=start, [2]=size
    static Tensor* split(Tensor* input, int dim, int64_t start, int64_t size, const std::string& name = "", int32_t layer_id = -1){
        Tensor* t = new Tensor;
        t->name = name;
        t->layer_id = layer_id;
        t->dtype = input->dtype;
        t->op_type = OperationType::OP_TYPE_SPLIT;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());
        t->dims[dim] = size;
        t->op_params[0] = static_cast<float>(dim);
        t->op_params[1] = static_cast<float>(start);
        t->op_params[2] = static_cast<float>(size);
        t->src[0] = input;
        OpFactory::compute_strides(t);
        return t;
    }
    // [X,Z] = [X,Y] * [Y,Z]
    static Tensor* mul(
        Tensor* a, 
        Tensor* b, 
        const std::string& name = "", 
        int32_t layer_id = -1
    ) {
        for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
            if (a->dims[i] != b->dims[i] && a->dims[i] > 0 && b->dims[i] > 0) {
                LOG_ERROR("mul: shape mismatch");
                throw std::runtime_error("mul: shape mismatch");
            }
        }
        Tensor* t = new Tensor();
        t->name = name.empty() ? a->name + "_mul_" + b->name : name;
        t->layer_id = layer_id;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_MUL;
        t->dtype = a->dtype;
        std::copy(a->dims.begin(), a->dims.end(), t->dims.begin());
        t->src[0] = a;
        t->src[1] = b;
        t->data = nullptr;  // 执行时分配
        t->offset = 0;
        OpFactory::compute_strides(t);
        return t;
    }
    // 
    static Tensor* add(Tensor* a, Tensor* b, const std::string& name = "", int32_t layer_id = -1){
        Tensor* t = new Tensor;
        t->name = name;
        t->layer_id = layer_id;
        t->dtype = a->dtype;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_ADD;
        std::copy(a->dims.begin(), a->dims.end(), t->dims.begin());
        t->src[0] = a;
        t->src[1] = b;
        OpFactory::compute_strides(t);
        return t;
    }
    // [B,M,N] -> [B,M,X,Y]; (X*Y = N)
    static Tensor* reshape(
        Tensor* input,
        std::initializer_list<int64_t> new_shape_init,
        const std::string& name = "",
        int32_t layer_id = -1
    ){
        std::vector<int64_t> new_shape(new_shape_init);
        
        // 收集源有效维度 (跳过 0)
        std::vector<int64_t> src_dims;
        for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
            if (input->dims[i] != 0) {
                src_dims.push_back(input->dims[i]);
            }
        }
        size_t src_elements = 1;
        bool src_has_dynamic = false;
        for (auto d : src_dims) {
            if (d == -1) {
                src_has_dynamic = true;  // 标记有动态维度
            } else {
                src_elements *= static_cast<size_t>(d);
            }
        }
        // 处理目标的 -1
        size_t known_elements = 1;
        int infer_idx = -1;
        for (size_t i = 0; i < new_shape.size(); ++i) {
            if (new_shape[i] == -1) {
                infer_idx = static_cast<int>(i);
            } else {
                known_elements *= static_cast<size_t>(new_shape[i]);
            }
        }
        if (infer_idx >= 0) {
            if (src_has_dynamic) {
                new_shape[infer_idx] = -1;  // 保持动态
            } else {
                if (src_elements % known_elements != 0) {
                    LOG_ERROR(std::format("Reshape invalid: {} elements cannot fit into shape", src_elements));
                    throw std::runtime_error(
                        std::format("Reshape invalid: {} elements cannot fit into shape", src_elements));
                }
                new_shape[infer_idx] = static_cast<int64_t>(src_elements / known_elements);
            }
        }
        // 验证 (跳过动态维度)
        if (!src_has_dynamic) {
            size_t target_elements = 1;
            for (auto d : new_shape) {
                if (d > 0) target_elements *= static_cast<size_t>(d);
            }
            if (src_elements != target_elements) {
                LOG_ERROR(std::format("Reshape mismatch: src={} vs target={}", src_elements, target_elements));
                throw std::runtime_error(
                    std::format("Reshape mismatch: src={} vs target={}", src_elements, target_elements));
            }
        }
        // 创建视图 Tensor
        Tensor* view = new Tensor();
        view->name = name.empty() ? input->name + "_reshape" : name;
        view->dtype = input->dtype;
        view->layer_id = layer_id;
        view->type = TensorType::TENSOR_TYPE_VIEW;
        view->op_type = OperationType::OP_TYPE_RESHAPE;
        std::fill(view->dims.begin(), view->dims.end(), 0);
        std::copy(new_shape.begin(), new_shape.end(), view->dims.begin());
        // 共享内存 (零拷贝)
        view->data = input->data;
        view->offset = input->offset;
        view->src[0] = input;
        OpFactory::compute_strides(view);
        return view;
    }
    static Tensor* permute(
        Tensor* input,
        std::initializer_list<int> perm_init,
        const std::string& name = "",
        int32_t layer_id = -1
    ){
        if (!input) return nullptr;
        std::vector<int> perm(perm_init);
        // 1. 收集输入有效维度
        std::vector<int64_t> src_dims;
        for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
            if (input->dims[i] != 0) {
                src_dims.push_back(input->dims[i]);
            }
        }
        int src_ndim = static_cast<int>(src_dims.size());
        // 2. 验证 perm 范围
        if (perm.size() != static_cast<size_t>(src_ndim)) {
            LOG_ERROR(std::format("Permute dim mismatch: perm.size()={} vs src_ndim={}", perm.size(), src_ndim));
            throw std::runtime_error(std::format("Permute dim mismatch: perm.size()={} vs src_ndim={}", perm.size(), src_ndim));
        }
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] < 0 || perm[i] >= src_ndim) {
                LOG_ERROR(std::format("Permute index {} out of range [0, {})", perm[i], src_ndim));
                throw std::runtime_error(std::format("Permute index {} out of range [0, {})", perm[i], src_ndim));
            }
        }

        Tensor* t = new Tensor();
        t->data = nullptr;
        t->name = name.empty() ? input->name + "_permute" : name;
        t->dtype = input->dtype;
        t->layer_id = layer_id;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_PERMUTE;
        std::fill(t->dims.begin(), t->dims.end(), 0);
        std::fill(t->op_params.begin(), t->op_params.end(), 0.0f);
        for (size_t i = 0; i < perm.size(); ++i) {
            t->dims[i] = src_dims[perm[i]];
            t->op_params[i] = static_cast<float>(perm[i]); // save perm for kernel
        }
        t->src[0] = input;
        OpFactory::compute_strides(t);
        return t;
    }
    static Tensor* reshape_permute(
        Tensor* input,
        std::initializer_list<int64_t> new_shape_init,
        std::initializer_list<int> new_perm_init,
        const std::string& name = "",
        int32_t layer_id = -1
    ){
        Tensor* reshaped = OpFactory::reshape(input, new_shape_init, name+"_reshape", layer_id);
        if (!reshaped) return nullptr;
        Tensor* result = OpFactory::permute(reshaped, new_perm_init, name+"_permute", layer_id);
        return result;
    }
    static Tensor* permute_reshape(
        Tensor* input,
        std::initializer_list<int> new_perm_init,
        std::initializer_list<int64_t> new_shape_init,
        const std::string& name = "",
        int32_t layer_id = -1
    ){
        Tensor* result = OpFactory::permute(input, new_perm_init, name+"_permute", layer_id);
        if (!result) 
            return nullptr;
        Tensor* reshaped = OpFactory::reshape(result, new_shape_init, name+"_reshape", layer_id);
        return reshaped;
    }
    static Tensor* repeat_kv(Tensor* kv, int n_rep, const std::string& name = ""){
        // todo...
        LOG_ERROR("repeat_kv not implemented yet");
        throw std::runtime_error("repeat_kv not implemented yet");
    }
    static Tensor* Attention(
        Tensor* q_rope, 
        Tensor* k_rope, 
        Tensor* v_4d,
        Tensor* mask = nullptr,
        float scale = -1.0f,
        bool causal = true,
        int num_kv_groups = 1,        // ⚠️ GQA: n_heads / n_kv_heads
        const std::string& name = "",
        OperationType optype = OperationType::OP_TYPE_PAGED_ATTN,
        int32_t layer_id = -1
    ){
        if (!q_rope || !k_rope || !v_4d) {
            LOG_ERROR("sdpa: null input tensor");
            throw std::runtime_error("sdpa: null input tensor");
        }
        // 验证维度
        int64_t n_heads = q_rope->dims[1];
        int64_t n_kv_heads = k_rope->dims[1];
        if (num_kv_groups < 1) {
            num_kv_groups = static_cast<int>(n_heads / n_kv_heads);
        }
        if (n_heads % n_kv_heads != 0) {
            LOG_ERROR(std::format("sdpa: n_heads={} must be divisible by n_kv_heads={}", n_heads, n_kv_heads));
            throw std::runtime_error(std::format("sdpa: n_heads={} must be divisible by n_kv_heads={}", n_heads, n_kv_heads));
        }
        // 创建输出 Tensor
        Tensor* t = new Tensor();
        t->name = name.empty() ? "sdpa_out" : name;
        t->dtype = q_rope->dtype;
        t->layer_id = layer_id;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = optype;
        
        // 输出形状同 Q: [B, n_heads, S, head_dim]
        std::copy(q_rope->dims.begin(), q_rope->dims.end(), t->dims.begin());
        
        // 视图：共享内存 (实际计算在后端)
        t->data = q_rope->data;  // ⚠️ 后端可能新分配
        t->offset = q_rope->offset;
        
        // 依赖追踪
        t->src[0] = q_rope;
        t->src[1] = k_rope;
        t->src[2] = v_4d;
        t->src[3] = mask;
        
        // 步长同 Q
        std::copy(q_rope->strides.begin(), q_rope->strides.end(), t->strides.begin());
        
        // 存储元数据到 op_params
        int64_t head_dim = q_rope->dims[3];
        int32_t head_dim_i32 = static_cast<int32_t>(head_dim);
        int32_t kv_groups_i32 = static_cast<int32_t>(num_kv_groups);
        float scale_val = (scale < 0) ? (1.0f / std::sqrt(static_cast<float>(head_dim))) : scale;

        int32_t causal_i32 = causal ? 1 : 0;
        
        t->op_params[0] = head_dim_i32;
        t->op_params[1] = scale_val;
        t->op_params[2] = causal_i32;
        t->op_params[3] = kv_groups_i32;
        return t;
    }
    // (q,sin,cos) -> q_rope
    // (k,sin,cos) -> k_rope
    static std::tuple<Tensor*, Tensor*> apply_rope(
        Tensor* q,                    // Query: [B, n_heads, seq, head_dim]
        Tensor* k,                    // Key:   [B, n_heads, seq, head_dim]
        Tensor* cos_cache,            // [max_seq, head_dim]
        Tensor* sin_cache,            // [max_seq, head_dim]
        Tensor* position_ids = nullptr,  // optional [1, seq]
        const std::string& name_suffix = "",
        int32_t layer_id = -1
    ){
        if (!q || !k || !cos_cache || !sin_cache) {
            LOG_ERROR("apply_rope: nullptr input tensor");
            throw std::runtime_error("apply_rope: nullptr input tensor");
        }
        // 提取 Q/K 的 head_dim (最后一个维度)
        int64_t head_dim = 0;
        for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
            if (q->dims[i] > 0) {
                head_dim = q->dims[i];
                break;
            }
        }
        if (head_dim <= 0 || head_dim % 2 != 0) {
            LOG_ERROR("apply_rope: head_dim must be positive even number");
            throw std::runtime_error("apply_rope: head_dim must be positive even number");
        }
        int64_t rope_dim = cos_cache->dims[1];
        if (rope_dim > head_dim || rope_dim % 2 != 0) {
            LOG_ERROR(std::format("apply_rope: invalid cache dim. cache_dim={}, head_dim={}", rope_dim, head_dim));
            throw std::runtime_error(std::format(
                "apply_rope: invalid cache dim. cache_dim={}, head_dim={}", rope_dim, head_dim));
        }
        int32_t head_dim_i32 = static_cast<int32_t>(head_dim);
        int32_t rope_dim_i32 = static_cast<int32_t>(rope_dim);
        auto make_rope_output = [&](Tensor* input, const std::string& name) -> Tensor* {
            Tensor* t = new Tensor();
            t->name = name;
            t->dtype = input->dtype;
            t->type = TensorType::TENSOR_TYPE_ACTIVATION;
            t->op_type = OperationType::OP_TYPE_APPLY_ROPE;
            // 形状与输入相同
            std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());
            t->data = input->data;
            t->offset = input->offset;
            t->layer_id = layer_id;
            t->src[0] = input;           // Q 或 K
            t->src[1] = cos_cache;       // cos 缓存
            t->src[2] = sin_cache;       // sin 缓存
            t->src[3] = position_ids;    // 可选位置 ID
            std::copy(input->strides.begin(), input->strides.end(), t->strides.begin());
            t->op_params[0] = head_dim_i32;
            t->op_params[1] = rope_dim_i32;
            t->op_params[2] = 0;    // 当前token的位置信息
            return t;
        };
        std::string suffix = name_suffix.empty() ? "" : "_" + name_suffix;
        Tensor* q_rope = make_rope_output(q, q->name + "_rope" + suffix);
        Tensor* k_rope = make_rope_output(k, k->name + "_rope" + suffix);
        return {q_rope, k_rope};
    }
    static void compute_strides(Tensor* t){
        size_t stride = 1;
        for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
            if (t->dims[i] == 0) {
                t->strides[i] = 0;
            } else {
                t->strides[i] = stride;
                stride *= t->dims[i];
            }
        }
    }
    
    static const TensorInfo* find_tensor(const GGUFInfo& info, const std::string& name){
        for (const auto& t : info.tensors_info) {
            if (t.name == name) {
                return &t;
            }
        }
        LOG_ERROR(std::format("Tensor not found: {}", name));
        throw std::runtime_error(std::format("Tensor not found: {}", name));
    }
    // ──────────────────────────────────────────────────────────────────
    // causal_conv1d: 因果一维卷积 (Mamba2 输入投影后)
    // src[0]=input, src[1]=weight, src[2]=conv_state (CACHE, decode用)
    // ──────────────────────────────────────────────────────────────────
    static Tensor* causal_conv1d(
        Tensor* input,
        const TensorInfo* weight_info,
        int kernel_size,
        const std::string& name = "",
        int32_t layer_id = -1
    ){
        Tensor* t = new Tensor();
        t->name = name;
        t->layer_id = layer_id;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->dtype = input->dtype;
        t->op_type = OperationType::OP_TYPE_CAUSAL_CONV1D;

        // 输出形状同输入 (causal padding = kernel-1, stride=1)
        std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());

        t->src[0] = input;
        t->src[1] = OpFactory::weight_placeholder(weight_info, weight_info->name, layer_id);

        // conv_state: 存最近 (kernel_size-1) 行输入，用于 decode 阶段
        int64_t d_inner = 0;
        for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
            if (input->dims[i] != 0) { d_inner = input->dims[i]; break; }
        }
        t->src[2] = OpFactory::placeholder(
            input->dtype, TensorType::TENSOR_TYPE_CACHE,
            {1, static_cast<int64_t>(kernel_size - 1), d_inner},
            name.empty() ? "conv_state" : name + "_state", layer_id
        );

        t->op_params[0] = static_cast<float>(kernel_size);

        OpFactory::compute_strides(t);
        return t;
    }

    // ──────────────────────────────────────────────────────────────────
    // ssm_scan: Gated Delta Rule 线性注意力 (Qwen3.5 SSM 层)
    // 对应 Python: torch_recurrent_gated_delta_rule(Q, K, V, g, beta)
    // g = -A_log.exp() * softplus(a_proj + dt_bias), beta = sigmoid(b_proj)
    // src[0]=Q [B,L,H_k,D], src[1]=K [B,L,H_k,D], src[2]=V [B,L,H_v,D],
    // src[3]=A_log [H_v] (per V head), src[4]=b_proj [B,L,H_v], src[5]=a_proj [B,L,H_v],
    // src[6]=dt_bias [H_v], src[7]=state [1,H_v,D,D] (CACHE, per V head)
    // op_params[0]=num_k_heads, op_params[1]=head_dim(D), op_params[2]=num_v_heads
    // ──────────────────────────────────────────────────────────────────
    static Tensor* ssm_scan(
        Tensor* q,                  // Q [B, L, H_k, D]
        Tensor* k,                  // K [B, L, H_k, D]
        Tensor* v,                  // V [B, L, H_v, D]
        const TensorInfo* a_info,   // A_log weight [H_k]
        Tensor* b_proj,             // beta logit [B, L, H_v]
        Tensor* a_proj,             // alpha/dt [B, L, H_v]
        const TensorInfo* dt_info,  // dt_bias weight [H_v]
        int64_t num_k_heads,
        int64_t num_v_heads,
        int64_t state_size,           // D (head_dim)
        const std::string& name = "",
        int32_t layer_id = -1
    ){
        Tensor* t = new Tensor();
        t->name = name;
        t->layer_id = layer_id;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->dtype = q->dtype;
        t->op_type = OperationType::OP_TYPE_SSM_SCAN;

        // 输出形状同 V: [B, L, H_v, D]
        std::copy(v->dims.begin(), v->dims.end(), t->dims.begin());

        t->src[0] = q;
        t->src[1] = k;
        t->src[2] = v;
        t->src[3] = OpFactory::weight_placeholder(a_info, a_info->name, layer_id); // 需要先 log(-A)
        t->src[4] = b_proj;
        t->src[5] = a_proj;
        t->src[6] = OpFactory::weight_placeholder(dt_info, dt_info->name, layer_id);

        // ssm_state: 每个 V head 一个独立的 D×D 状态 [1, H_v, D, D]
        t->src[7] = OpFactory::placeholder(
            DataType::GGML_TYPE_F32, TensorType::TENSOR_TYPE_CACHE,
            {1, num_v_heads, state_size, state_size},
            name.empty() ? "ssm_state" : name + "_state", layer_id
        );

        t->op_params[0] = static_cast<float>(num_k_heads);
        t->op_params[1] = static_cast<float>(state_size);
        t->op_params[2] = static_cast<float>(num_v_heads);

        OpFactory::compute_strides(t);
        return t;
    }
    static std::array<int64_t, TENSOR_MAX_DIMS> infer_linear_output(
        std::span<const int64_t> input_dims, 
        std::span<const int64_t> weight_dims,
        bool transpose
    ) {
        std::array<int64_t, TENSOR_MAX_DIMS> out{};
        std::fill(out.begin(), out.end(), 0);
        int input_ndim = 0;
        for (int i = static_cast<int>(input_dims.size()) - 1; i >= 0; --i) {
            if (input_dims[i] != 0) {
                input_ndim = i + 1;
                break;
            }
        }
        if (input_ndim == 0) {
            LOG_ERROR("Empty input shape");
            throw std::runtime_error("Empty input shape");
        }
        int64_t weight_out = weight_dims[0]; 
        int64_t weight_in = weight_dims[1];
        if (!transpose) {
            std::swap(weight_out, weight_in);
        }
        int64_t input_last = input_dims[input_ndim - 1];
        if (input_last > 0 && weight_in > 0 && input_last != weight_in) {
            LOG_ERROR(std::format("Linear dimension mismatch: input last={} vs weight in={}", input_last, weight_in));
            throw std::runtime_error(std::format("Linear dimension mismatch: input last={} vs weight in={}",input_last, weight_in));
        }
        for (int i = 0; i < input_ndim - 1; ++i) {
            out[i] = input_dims[i];
        }
        out[input_ndim - 1] = weight_out;
        return out;
    }
};