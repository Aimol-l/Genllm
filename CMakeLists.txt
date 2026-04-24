cmake_minimum_required(VERSION 3.28)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THIS_PROJECT_NAME "gnellm")

option(BUILD_TEST      "build test file with ./tests/*.cpp" OFF)

option(BACKEND_CPU     "Enable CPU backend" OFF)
option(BACKEND_CUDA    "Enable CUDA backend" OFF)
option(BACKEND_SYCL    "Enable SYCL backend" OFF)
option(BACKEND_VULKAN  "Enable Vulkan backend" OFF)

# 检查是否启用了至少一个后端
if(NOT BACKEND_CPU AND NOT BACKEND_CUDA AND NOT BACKEND_SYCL AND NOT BACKEND_VULKAN)
    message(WARNING "No backend enabled. Please enable at least one backend (BACKEND_CPU, BACKEND_CUDA, BACKEND_SYCL, or BACKEND_VULKAN). Enabling CPU backend by default.")
    set(BACKEND_CPU ON)
endif()



project(
    ${THIS_PROJECT_NAME}
    LANGUAGES C CXX CUDA
    DESCRIPTION "gguf inference tool"
    VERSION 0.1.0
)

# CPU配置，avx2，openmp,fma
if(BACKEND_CPU)
    find_package(OpenMP REQUIRED)
    add_definitions(-DBACKEND_CPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mf16c -mfma -fopenmp -fno-omit-frame-pointer -g")
endif()

# cuda配置,avx2,openmp,fma
if(BACKEND_CUDA)
    add_definitions(-DBACKEND_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    find_package(OpenMP REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_ARCHITECTURES "native")  # 常见现代GPU架构
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mf16c -mfma -fopenmp -fno-omit-frame-pointer -g")
endif()

# vk配置,avx2,openmp,fma
if(BACKEND_VULKAN)
    add_definitions(-DBACKEND_VULKAN)
    find_package(Vulkan REQUIRED)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -mavx2 -mf16c -mfma -fopenmp -fno-omit-frame-pointer -g")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)


# ------------------------- 必要头文件目录 -------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/3rd
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/backend
    ${PROJECT_SOURCE_DIR}/include/model
    ${PROJECT_SOURCE_DIR}/include/utils
)
# 后端头文件目录
if(BACKEND_CPU)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/cpu)
endif()
if(BACKEND_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/cuda)
endif()
if(BACKEND_SYCL)
    find_package(IntelSYCL REQUIRED)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/sycl)
endif()
if(BACKEND_VULKAN)
    include_directories(${CMAKE_SOURCE_DIR}/include/backend/cpu)
    include_directories(${PROJECT_SOURCE_DIR}/include/backend/vulkan)
endif()


# ------------------------- 源文件收集 -------------------------
file(GLOB SRC_FILES
    CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/core/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/model/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/backend/cpu/*.cpp" # CPU后端总是加载
    "${PROJECT_SOURCE_DIR}/src/backend/backend_cpu.cpp"
)
# 各个后端的源文件
if(BACKEND_CUDA)
    file(GLOB_RECURSE CUDA_OPS_SOURCES 
        "${PROJECT_SOURCE_DIR}/src/backend/cuda/*.cu"
        "${PROJECT_SOURCE_DIR}/src/backend/backend_cuda.cpp"
    )
    list(APPEND SRC_FILES ${CUDA_OPS_SOURCES})
endif()
if(BACKEND_SYCL)
    file(GLOB_RECURSE SYCL_OPS_SOURCES 
        "${PROJECT_SOURCE_DIR}/src/backend/sycl/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/backend/backend_sycl.cpp"
    )
    list(APPEND SRC_FILES ${SYCL_OPS_SOURCES})

endif()
if(BACKEND_VULKAN)
    file(GLOB_RECURSE VULKAN_OPS_SOURCES 
        "${PROJECT_SOURCE_DIR}/src/backend/vulkan/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/backend/backend_vulkan.cpp"
    )
    list(APPEND SRC_FILES ${VULKAN_OPS_SOURCES})
endif()

# ------------------------- 主库构建 -------------------------
add_library(${PROJECT_NAME} SHARED ${SRC_FILES})

# 设置库的属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# 链接库
if(BACKEND_CPU AND OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

if(BACKEND_CUDA AND CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

if(BACKEND_SYCL AND CMAKE_SYCL_ENABLED)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${IntelSYCL_LIBRARIES})
endif()

if(BACKEND_VULKAN AND Vulkan_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${Vulkan_LIBRARIES})
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# 设置库的别名
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# ------------------------- 安装配置 -------------------------

include(GNUInstallDirs)
# 安装目标库
install(TARGETS ${PROJECT_NAME}
    EXPORT GenllmTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/include/3rd/json.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/3rd
)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/include/core/executor.h
        ${PROJECT_SOURCE_DIR}/include/core/gguf_parser.h
        ${PROJECT_SOURCE_DIR}/include/core/graph.h
        ${PROJECT_SOURCE_DIR}/include/core/kernels.h
        ${PROJECT_SOURCE_DIR}/include/core/manager.h
        ${PROJECT_SOURCE_DIR}/include/core/pools.h
        ${PROJECT_SOURCE_DIR}/include/core/resource.h
        ${PROJECT_SOURCE_DIR}/include/core/scheduler.h
        ${PROJECT_SOURCE_DIR}/include/core/tensor.hpp
        ${PROJECT_SOURCE_DIR}/include/core/thread_pool.h
        ${PROJECT_SOURCE_DIR}/include/core/tokenizer.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/core
)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/include/model/model.h
        ${PROJECT_SOURCE_DIR}/include/model/op_factory.hpp
        ${PROJECT_SOURCE_DIR}/include/model/Qwen3.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/model
)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/include/utils/bfloat16.hpp
        ${PROJECT_SOURCE_DIR}/include/utils/dtype_traits.hpp
        ${PROJECT_SOURCE_DIR}/include/utils/float16.hpp
        ${PROJECT_SOURCE_DIR}/include/utils/tools.hpp
        ${PROJECT_SOURCE_DIR}/include/utils/utils.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/utils
)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/include/backend/backend.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/backend
)
# 安装后端头文件
if(BACKEND_CPU)
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/backend/cpu/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/backend/cpu
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endif()
if(BACKEND_CUDA)
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/backend/cuda/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/backend/cuda
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
    )
endif()
if(BACKEND_SYCL)
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/backend/sycl/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/backend/sycl
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endif()
if(BACKEND_VULKAN)
    install(
        DIRECTORY ${PROJECT_SOURCE_DIR}/include/backend/vulkan/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/genllm/backend/vulkan
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
endif()
# 设置CMake包配置
install(EXPORT GenllmTargets
    FILE GenllmTargets.cmake
    NAMESPACE Genllm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/genllm
)
# 配置包配置文件
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/Genllm.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/GenllmConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/genllm
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GenllmConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/genllm
)

# ------------------------- 测试系统 -------------------------
if(BUILD_TEST)
    enable_testing()
    file(GLOB TEST_FILES "${PROJECT_SOURCE_DIR}/tests/*.cpp")
    foreach(test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} PRIVATE ${PROJECT_NAME})
        target_include_directories(${test_name} PRIVATE ${INCLUDE_DIRS})
        # 为测试目标添加后端定义
        if(BACKEND_CPU)
           target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(BACKEND_SYCL)
            target_compile_options(${test_name} PRIVATE -fsycl -O2 -fiopenmp -mavx2 -mf16c -mfma -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64)
            target_link_libraries(${test_name} PRIVATE ${IntelSYCL_LIBRARIES})
        endif()
        if(BACKEND_CUDA)
           target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        if(BACKEND_VULKAN)
            target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${Vulkan_LIBRARIES})
            target_link_libraries(${test_name} PRIVATE OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()


