cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(PROJECT_NAME "cuda_project")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "~/libs")
list(APPEND CMAKE_PREFIX_PATH "~/libs")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/build")

option(USE_LOCAL_RENDERER "Builds the project for pro-viz use" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
cmake_policy(SET CMP0074 NEW)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)
find_package(Threads REQUIRED)
find_package(protobuf 5.27.0 CONFIG REQUIRED)
find_package(assimp 5.4.3 REQUIRED)
find_package(ixwebsocket REQUIRED)
find_package(libx264 REQUIRED)
find_package(libjpeg-turbo REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
                -gencode arch=compute_70,code=sm_70
                -rdc=true
                -lineinfo
)

add_library(proto-objects OBJECT "${CMAKE_CURRENT_LIST_DIR}/proto/main.proto")
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
cuda_add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} assimp::assimp CUDA::nvml ixwebsocket::ixwebsocket PNG::PNG glm::glm protobuf::libprotobuf libx264::libx264 libjpeg-turbo::libjpeg-turbo proto-objects Threads::Threads)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75,70")

# This shouldn't be necessary...
include_directories(${protobuf_INCLUDE_DIRS})
include_directories(${absl_INCLUDE_DIRS})

# Generate protobuf files
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_LIST_DIR}/src/genproto")
target_include_directories(proto-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")
protobuf_generate(
   TARGET proto-objects
   IMPORT_DIRS "${CMAKE_CURRENT_LIST_DIR}/proto"
   PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

if (USE_LOCAL_RENDERER)
    set(OpenGL_GL_PREFERENCE GLVND)

    find_package(glfw3 REQUIRED CONFIG)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)

    target_link_libraries(${PROJECT_NAME} glfw OpenGL::GL GLEW::GLEW)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_LOCAL_RENDERER)
endif()
