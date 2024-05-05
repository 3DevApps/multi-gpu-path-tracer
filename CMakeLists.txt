cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(PROJECT_NAME "cuda_project")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "~/libs")
list(APPEND CMAKE_PREFIX_PATH "~/libs")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(OpenGL_GL_PREFERENCE GLVND)
cmake_policy(SET CMP0074 NEW)


find_package(CUDA REQUIRED)
find_package(glfw3 REQUIRED CONFIG)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(assimp REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
        CUDA_NVCC_FLAGS 
        ${CUDA_NVCC_FLAGS}; 
                -gencode arch=compute_75,code=sm_75
                -gencode arch=compute_70,code=sm_70 
                -rdc=true
)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
cuda_add_executable(cuda_project ${SOURCES})

target_link_libraries(cuda_project glfw OpenGL::GL GLEW::GLEW assimp::assimp)
set_target_properties(cuda_project PROPERTIES CUDA_ARCHITECTURES "75,70")
