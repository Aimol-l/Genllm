cmake_minimum_required(VERSION 3.28)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(THIS_PROJECT_NAME "main")

project(
    ${THIS_PROJECT_NAME}
    LANGUAGES C CXX
    DESCRIPTION "test for gguf parser"
)

option(BACKEND_CUDA "Enable CUDA backend support" OFF)

file(GLOB SRC_FILES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*/*.cpp"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/core
    ${PROJECT_SOURCE_DIR}/include/backend
    ${PROJECT_SOURCE_DIR}/include/model
)

if(BACKEND_CUDA)
    include(FetchContent)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found. Install CUDA toolkit or disable BACKEND_CUDA.")
    endif()
    enable_language(CUDA)

    find_package(CUDAToolkit REQUIRED)

    list(FILTER SRC_FILES EXCLUDE REGEX "backend_cuda\\.cpp$")
    add_executable(${THIS_PROJECT_NAME} ${SRC_FILES} "${PROJECT_SOURCE_DIR}/src/backend/backend_cuda.cpp")

    target_compile_definitions(${THIS_PROJECT_NAME} PRIVATE BACKEND_CUDA)
    target_link_libraries(${THIS_PROJECT_NAME} PRIVATE CUDA::cudart)
else()
    list(FILTER SRC_FILES EXCLUDE REGEX "backend_cuda\\.cpp$")
    add_executable(${THIS_PROJECT_NAME} ${SRC_FILES})
endif()
