// model.cpp - 模型工厂实现
#include "model/Qwen3.hpp"
#include "model/Qwen35.hpp"


// 初始化默认模型注册
void ModelFactory::init() {
    RegisterModel<Qwen3Model>("qwen3");
    // RegisterModel<Qwen35Model>("qwen35"); // 暂时未完成
}

// 自动初始化
namespace {
    struct ModelFactoryInitializer {
        ModelFactoryInitializer() {
            ModelFactory::init();
        }
    };
    static ModelFactoryInitializer g_model_factory_init;
}
