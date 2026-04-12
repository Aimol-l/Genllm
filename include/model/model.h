#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <format>
#include <stdexcept>
#include <memory>
#include <algorithm>
#include "graph.hpp"
#include "gguf_parser.h"
// #include "core/mmanger.h"
#include "utils/utils.hpp"


// 模型基类
class ModelBase {
protected:
    enum ModelType type;
    enum ModelArch arch;
public:
    std::string name = "unknow";
    ComputeGraph graph_;
    virtual ~ModelBase(){
        // todo...
    }
    virtual void print_info() = 0;
    virtual void set_params(void*) = 0;
    virtual ComputeGraph& build_graph(const GGUFInfo&) = 0;
    virtual int64_t max_seq_len() const { return 1; }
};


// 模型工厂类
class ModelFactory {
private:
    // 注册的模型创建函数
    using ModelCreator = std::unique_ptr<ModelBase> (*)();
    static inline std::unordered_map<std::string, ModelCreator> s_registry;

    // 注册模型的辅助函数
    template<typename T>
    static std::unique_ptr<ModelBase> create_model() {
        return std::make_unique<T>();
    }

public:
    // 注册模型类型
    template<typename T>
    static void RegisterModel(const std::string& arch_name) {
        s_registry[arch_name] = &create_model<T>;
    }

    // 根据架构名称创建模型
    [[nodiscard]] static std::unique_ptr<ModelBase> CreateFromArch(const std::string& arch_name) {
        auto it = s_registry.find(arch_name);
        if (it == s_registry.end()) {
            throw std::runtime_error(std::format("Unsupported model architecture: {}", arch_name));
        }
        return it->second();
    }

    // 从GGUFInfo创建模型
    [[nodiscard]] static std::unique_ptr<ModelBase> CreateFromGGUF(GGUFInfo& info) {
        std::string arch = info.get_model_architecture();
        // 转换为小写以匹配
        std::transform(arch.begin(), arch.end(), arch.begin(),[](unsigned char c){ return std::tolower(c); });
        return CreateFromArch(arch);
    }

    // 获取已注册的模型列表
    [[nodiscard]] static std::vector<std::string> registered_models() {
        std::vector<std::string> models;
        for (const auto& [name, _] : s_registry) {
            models.push_back(name);
        }
        return models;
    }

    // 初始化默认模型注册
    static void init();
};

// 全局模型注册宏
#define REGISTER_MODEL(ARCH_NAME, MODEL_CLASS) \
    namespace { \
        struct MODEL_CLASS##Registrar { \
            MODEL_CLASS##Registrar() { \
                ModelFactory::register_model<MODEL_CLASS>(ARCH_NAME); \
            } \
        }; \
        static MODEL_CLASS##Registrar g_##MODEL_CLASS##_registrar; \
    }

