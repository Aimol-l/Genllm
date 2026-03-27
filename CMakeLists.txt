cmake_minimum_required(VERSION 3.20)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()
set(CMAKE_CXX_STANDARD 23) # 设置为C++ 20标准
set(CMAKE_CXX_STANDARD_REQUIRED ON) #强制C++20
set(CMAKE_CXX_EXTENSIONS OFF)
set(THIS_PROJECT_NAME "main")

project(
    ${THIS_PROJECT_NAME}        #项目名称
    LANGUAGES C CXX      #编程语言
    DESCRIPTION "test for gguf parser"  #项目描述
)
file(GLOB SRC_FILES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*/*.cpp"

)
get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR}/bin)

include_directories(${PROJECT_DIR}/include)
include_directories(${PROJECT_DIR}/include/core)
include_directories(${PROJECT_DIR}/include/backend)


add_executable(${THIS_PROJECT_NAME}  ${SRC_FILES})
