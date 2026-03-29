#pragma once
#include <span>
#include <string>
#include "core/tensor.hpp"
#include "core/gguf_parser.h"


// ============================================================================
// OpFactory: 通用算子节点创建 (只建图，不分配内存)
// 所有函数返回新 Tensor*，用户负责生命周期管理
// ============================================================================
class OpFactory {
public:
    // 分配到权重区，但不是权重
    static Tensor* placeholder(
        DataType dtype,
        TensorType type, 
        std::initializer_list<int64_t> dims, 
        const std::string& name
    ){
        if(dims.size()>4){
            throw std::runtime_error("dims size must 1~4");
        }
        Tensor* t = new Tensor();
        t->name = name;
        t->dtype = dtype;
        t->type = type;
        t->op_type = OperationType::OP_TYPE_NONE;
        std::copy(dims.begin(), dims.end(), t->dims.begin());
        OpFactory::compute_strides(t);
        t->data = nullptr;  // 等待外部绑定
        return t;
    }
    // 静态权重占位符 (从 GGUF 加载，data=nullptr，执行前绑定)
    static Tensor* weight_placeholder(const TensorInfo* info, const std::string& name){
        Tensor* t = new Tensor();
        t->name = name;
        t->dtype = info->dtype;
        t->type = TensorType::TENSOR_TYPE_WEIGHT;
        t->op_type = OperationType::OP_TYPE_NONE;
        std::copy(info->dimensions.begin(), info->dimensions.end(), t->dims.begin());
        OpFactory::compute_strides(t);
        t->data = nullptr;  // 等待外部绑定
        return t;
    }
    // ──────────────────────────────────────────────────────────────────
    // linear: y = x @ weight.T
    // ──────────────────────────────────────────────────────────────────
    static Tensor* linear(Tensor* input, const TensorInfo* weight_info,bool transpose = false,const std::string& name = ""){
        Tensor* t = new Tensor();
        t->name = name;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->dtype = input->dtype;  // 输出 dtype 同输入
        t->op_type = OperationType::OP_TYPE_LINEAR;
        auto out_dims = infer_linear_output(
            {input->dims.begin(), input->dims.end()},
            weight_info->dimensions,
            transpose
        );
        std::copy(out_dims.begin(), out_dims.end(), t->dims.begin());
        // 双输入: [0]=dynamic input, [1]=static weight
        t->src[0] = input;
        t->src[1] = weight_placeholder(weight_info, weight_info->name);
        OpFactory::compute_strides(t);
        return t;
    }

    static Tensor* embedding_lookup(
        Tensor* input_ids,
        const TensorInfo* weight_info,
        const std::string& name = ""
    ){
        Tensor* t = new Tensor();
        t->name = name;
        t->dtype = weight_info->dtype;  // eg:BF16
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_EMBEDDING;
        t->dims[0] = input_ids->dims[0];  // batch
        t->dims[1] = input_ids->dims[1];  // seq_len
        t->dims[2] = weight_info->dimensions[1];  // hidden_size
        t->src[0] = input_ids;
        t->src[1] = OpFactory::weight_placeholder(weight_info, weight_info->name);
        OpFactory::compute_strides(t);
        return t;
    }
    // ──────────────────────────────────────────────────────────────────
    // rms_norm: y = (x / sqrt(mean(x²)+eps)) * weight
    // ──────────────────────────────────────────────────────────────────
    static Tensor* rms_norm(
        Tensor* input,
        const TensorInfo* weight_info,
        float eps,
        const std::string& name = ""
    ){
        Tensor* t = new Tensor();
        t->name = name;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->dtype = input->dtype;
        t->op_type = OperationType::OP_TYPE_RMS_NORM;
        // 输出形状同输入
        std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());
        t->src[0] = input;
        t->src[1] = OpFactory::weight_placeholder(weight_info, weight_info->name);
        t->op_params[0] = eps;
        OpFactory::compute_strides(t);
        return t;
    }
    static std::tuple<Tensor*, Tensor*> rope_cache(
        int max_seq_len,
        int head_dim, 
        float theta, 
        DataType dtype, 
        const std::string& name_prefix = "rope"
    ){
        int half_dim = head_dim / 2;
        auto make_cache_tensor = [&](const std::string& name) -> Tensor* {
            Tensor* t = new Tensor();
            t->name = name;
            t->dtype = dtype;
            t->type = TensorType::TENSOR_TYPE_CACHE;
            t->op_type = OperationType::OP_TYPE_ROPE_CACHE;
            // 形状: [max_seq_len, half_dim]
            t->dims[0] = max_seq_len;
            t->dims[1] = half_dim;
            for (int i = 2; i < TENSOR_MAX_DIMS; ++i) t->dims[i] = 1;
            // 步长: 行优先 (字节跨度)
            size_t elem_bytes = data_type_size(dtype);
            t->strides[1] = elem_bytes;
            t->strides[0] = half_dim * elem_bytes;
            for (int i = 2; i < TENSOR_MAX_DIMS; ++i) t->strides[i] = 0;
            t->data = nullptr;
            t->offset = 0;
            t->backend = nullptr;
            return t;
        };
        Tensor* cos_tensor = make_cache_tensor(name_prefix + "_cos");
        Tensor* sin_tensor = make_cache_tensor(name_prefix + "_sin");
        sin_tensor->op_params[0] = theta;
        sin_tensor->op_params[1] = head_dim;
        sin_tensor->op_params[2] = max_seq_len;
        cos_tensor->op_params[0] = theta;
        cos_tensor->op_params[1] = head_dim;
        cos_tensor->op_params[2] = max_seq_len;
        return {sin_tensor,cos_tensor};
    }
    // y = x * sigmoid(x)
    static Tensor* silu(Tensor* input, const std::string& name = ""){
        Tensor* t = new Tensor;
        t->name = name;
        t->dtype = input->dtype;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;

        t->op_type = OperationType::OP_TYPE_SILU;
        std::copy(input->dims.begin(), input->dims.end(), t->dims.begin());
        t->src[0] = input;
        OpFactory::compute_strides(t);
        return t;
    }
    // [X,Z] = [X,Y] * [Y,Z]
    static Tensor* mul(Tensor* a, Tensor* b, const std::string& name = ""){
        Tensor* t = new Tensor;
        t->name = name;
        t->dtype = a->dtype;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;

        t->op_type = OperationType::OP_TYPE_MAT_MUL;
        t->dims[0] = a->dims[0];
        t->dims[1] = b->dims[1];
        t->src[0] = a;
        t->src[1] = b;
        OpFactory::compute_strides(t);
        return t;
    }
    // 
    static Tensor* add(Tensor* a, Tensor* b, const std::string& name = ""){
        Tensor* t = new Tensor;
        t->name = name;
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
        const std::string& name = ""
    ){
        std::vector<int64_t> new_shape(new_shape_init);
        std::vector<int64_t> src_dims;
        for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
            if (input->dims[i] > 0) {
                src_dims.push_back(input->dims[i]);
            }
        }
        // 2. 计算元素总数
        size_t src_elements = std::accumulate(src_dims.begin(),src_dims.end(),1,std::multiplies<int64_t>());
        // 3. 处理 -1: 自动推断
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
            if (src_elements % known_elements != 0) {
                throw std::runtime_error(std::format("Reshape invalid: {} elements cannot fit into shape", src_elements));
            }
            new_shape[infer_idx] = static_cast<int64_t>(src_elements / known_elements);
        }
        // 4. 验证元素数匹配
        size_t target_elements = std::accumulate(new_shape.begin(),new_shape.end(),1,std::multiplies<int64_t>());
        if (src_elements != target_elements) {
            throw std::runtime_error(std::format("Reshape mismatch: src={} vs target={}", src_elements, target_elements));
        }
        // 5. 创建视图 Tensor
        Tensor* view = new Tensor();
        view->name = name.empty() ? input->name + "_reshape" : name;
        view->dtype = input->dtype;
        view->type = TensorType::TENSOR_TYPE_VIEW;
        view->op_type = OperationType::OP_TYPE_RESHAPE;
        // 填充形状
        std::fill(view->dims.begin(), view->dims.end(), 1);
        std::copy(new_shape.begin(), new_shape.end(), view->dims.begin());
        // 共享内存 (零拷贝)
        view->data = input->data;
        view->offset = input->offset;
        view->backend = input->backend;
        view->device_id = input->device_id;
        view->src[0] = input;               // 依赖追踪
        OpFactory::compute_strides(view);
        return view;
    }
    static Tensor* permute(
        Tensor* input,
        std::initializer_list<int> perm_init,
        const std::string& name = ""
    ){
        if (!input) return nullptr;
        std::vector<int> perm(perm_init);
        // 1. 收集输入有效维度
        std::vector<int64_t> src_dims;
        for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
            if (input->dims[i] > 0) {
                src_dims.push_back(input->dims[i]);
            }
        }
        int src_ndim = static_cast<int>(src_dims.size());
        // 2. 验证 perm 范围
        if (perm.size() != static_cast<size_t>(src_ndim)) {
            throw std::runtime_error(std::format("Permute dim mismatch: perm.size()={} vs src_ndim={}", perm.size(), src_ndim));
        }
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] < 0 || perm[i] >= src_ndim) {
                throw std::runtime_error(std::format("Permute index {} out of range [0, {})", perm[i], src_ndim));
            }
        }

        Tensor* t = new Tensor();
        t->name = name.empty() ? input->name + "_permute" : name;
        t->dtype = input->dtype;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_PERMUTE;
        std::fill(t->dims.begin(), t->dims.end(), 1);
        for (size_t i = 0; i < perm.size(); ++i) {
            t->dims[i] = src_dims[perm[i]];
        }
        t->src[0] = input;
        OpFactory::compute_strides(t);
        return t;
    }
    static Tensor* reshape_permute(
        Tensor* input,
        std::initializer_list<int64_t> new_shape_init,
        std::initializer_list<int> perm_init,
        const std::string& name = ""
    ){
        Tensor* reshaped = OpFactory::reshape(input, new_shape_init, input->name + "_rp_reshape");
        if (!reshaped) 
            return nullptr;
        Tensor* result = OpFactory::permute(reshaped, perm_init, name);
        return result;
    }
    static Tensor* repeat_kv(Tensor* kv, int n_rep, const std::string& name = ""){

    }
    //Scaled Dot-Product Attention
    static Tensor* SDPA(
        Tensor* q_rope, 
        Tensor* k_rope, 
        Tensor* v_4d,
        Tensor* mask = nullptr,
        float scale = -1.0f,
        bool causal = true,
        int num_kv_groups = 1,        // ⚠️ GQA: n_heads / n_kv_heads
        const std::string& name = ""
    ){
        if (!q_rope || !k_rope || !v_4d) {
            throw std::runtime_error("sdpa: null input tensor");
        }
        // 验证维度
        int64_t n_heads = q_rope->dims[1];
        int64_t n_kv_heads = k_rope->dims[1];
        if (num_kv_groups < 1) {
            num_kv_groups = static_cast<int>(n_heads / n_kv_heads);
        }
        if (n_heads % n_kv_heads != 0) {
            throw std::runtime_error(std::format("sdpa: n_heads={} must be divisible by n_kv_heads={}", n_heads, n_kv_heads));
        }
        // 创建输出 Tensor
        Tensor* t = new Tensor();
        t->name = name.empty() ? "sdpa_out" : name;
        t->dtype = q_rope->dtype;
        t->type = TensorType::TENSOR_TYPE_ACTIVATION;
        t->op_type = OperationType::OP_TYPE_SDPA;
        
        // 输出形状同 Q: [B, n_heads, S, head_dim]
        std::copy(q_rope->dims.begin(), q_rope->dims.end(), t->dims.begin());
        
        // 视图：共享内存 (实际计算在后端)
        t->data = q_rope->data;  // ⚠️ 后端可能新分配
        t->offset = q_rope->offset;
        t->backend = q_rope->backend;
        t->device_id = q_rope->device_id;
        
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
        Tensor* q,                    // Query: [B, n_heads, S, head_dim]
        Tensor* k,                    // Key:   [B, n_heads, S, head_dim]
        Tensor* cos_cache,            // [max_seq, head_dim/2]
        Tensor* sin_cache,            // [max_seq, head_dim/2]
        Tensor* position_ids = nullptr,  // optional [1, S]
        const std::string& name_suffix = ""
    ){
        if (!q || !k || !cos_cache || !sin_cache) {
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
            throw std::runtime_error("apply_rope: head_dim must be positive even number");
        }
        int64_t half_dim = head_dim / 2;
        // 验证 cos/sin 缓存维度: [max_seq, half_dim]
        if (cos_cache->dims[1] != half_dim || sin_cache->dims[1] != half_dim) {
            throw std::runtime_error(std::format("apply_rope: cos/sin cache dim mismatch. expected half_dim={}, got {}", half_dim, cos_cache->dims[1]));
        }
        // 验证 Q/K 维度一致
        for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
            if (q->dims[i] != k->dims[i]) {
                throw std::runtime_error("apply_rope: Q and K must have same shape");
            }
        }
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
            t->backend = input->backend;
            t->device_id = input->device_id;
            t->src[0] = input;           // Q 或 K
            t->src[1] = cos_cache;       // cos 缓存
            t->src[2] = sin_cache;       // sin 缓存
            t->src[3] = position_ids;    // 可选位置 ID
            std::copy(input->strides.begin(), input->strides.end(), t->strides.begin());
            return t;
        };
        std::string suffix = name_suffix.empty() ? "" : "_" + name_suffix;

        Tensor* q_rope = make_rope_output(q, q->name + "_rope" + suffix);
        Tensor* k_rope = make_rope_output(k, k->name + "_rope" + suffix);

        int32_t head_dim_i32 = static_cast<int32_t>(head_dim);
        int32_t half_dim_i32 = static_cast<int32_t>(half_dim);
        
        q_rope->op_params[0] = head_dim_i32;
        q_rope->op_params[1] = half_dim_i32;

        k_rope->op_params[0] = head_dim_i32;
        k_rope->op_params[1] = half_dim_i32;
        return {q_rope, k_rope};
    }
    static void compute_strides(Tensor* t){
        size_t stride = 1;
        for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
            if (t->dims[i] == 0) {
                t->strides[i] = 0;
            } else {
                t->strides[i] = stride * data_type_size(t->dtype);
                stride *= t->dims[i];
            }
        }
    }
    
    static const TensorInfo* find_tensor(const GGUFInfo& info, const std::string& name) {
        for (const auto& t : info.tensors_info) {
            if (t.name == name) {
                return &t;
            }
        }
        throw std::runtime_error(std::format("Tensor not found: {}", name));
    }
    static std::array<int64_t, TENSOR_MAX_DIMS> 
    infer_linear_output(
        std::span<const int64_t> input_dims, 
        std::span<const int64_t> weight_dims,
        bool transpose) 
    {
        std::array<int64_t, TENSOR_MAX_DIMS> out{};
        std::fill(out.begin(), out.end(), 1);  // 初始化为 1
        if (input_dims.empty() || weight_dims.empty()) {
            throw std::runtime_error("infer_linear_output: empty input or weight dims");
        }
        if (weight_dims.size() != 2) {
            throw std::runtime_error(
                std::format("infer_linear_output: weight must be 2D, got {}D", weight_dims.size()));
        }
        // GGUF 权重布局: [out_features, in_features]
        int64_t weight_out = weight_dims[0];
        int64_t weight_in = weight_dims[1];
        if (transpose) {
            // 转置后: [in_features, out_features]
            std::swap(weight_out, weight_in);
        }
        // 输入最后一个维度必须匹配 weight_in
        int64_t input_last_dim = 0;
        for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
            if (input_dims[i] > 0) {
                input_last_dim = input_dims[i];
                break;
            }
        }
        if (input_last_dim > 0 && weight_in > 0 && input_last_dim != weight_in) {
            throw std::runtime_error(
                std::format("infer_linear_output: dimension mismatch. ""input last dim={} vs weight in_dim={}", input_last_dim, weight_in));
        }
        // 保留输入的前 N-1 个维度 (batch, seq_len, ...)
        int out_idx = 0;
        for (size_t i = 0; i < input_dims.size(); ++i) {
            if (input_dims[i] > 0 || i < input_dims.size() - 1) {
                // 非最后一个维度，或动态维度 (-1)
                out[out_idx++] = input_dims[i];
            }
        }
        // 最后一个维度替换为 weight_out
        // 找到输出中最后一个有效位置
        int last_valid_idx = 0;
        for (int i = TENSOR_MAX_DIMS - 1; i >= 0; --i) {
            if (out[i] > 0 || i < static_cast<int>(input_dims.size())) {
                last_valid_idx = i;
                break;
            }
        }
        // 更简单的方法: 直接覆盖输入的最后一个维度位置
        out_idx = 0;
        for (size_t i = 0; i < input_dims.size(); ++i) {
            if (i == input_dims.size() - 1) {
                // 最后一个维度: 替换为 out_features
                out[out_idx++] = weight_out;
            } else {
                // 其他维度: 保留
                out[out_idx++] = input_dims[i];
            }
        }
        return out;
    }
    static std::array<int64_t, TENSOR_MAX_DIMS> infer_linear_output(
        const Tensor* input, 
        const TensorInfo* weight_info,
        bool transpose)
    {
        std::vector<int64_t> input_dims;
        for (int i = 0; i < TENSOR_MAX_DIMS; ++i) {
            if (input->dims[i] > 0 || i < TENSOR_MAX_DIMS - 1) {
                input_dims.push_back(input->dims[i]);
            }
        }
        return infer_linear_output(input_dims, weight_info->dimensions, transpose);
    }
};