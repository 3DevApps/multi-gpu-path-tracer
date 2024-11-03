cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(PROJECT_NAME "cuda_project")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "~/libs")
list(APPEND CMAKE_PREFIX_PATH "~/libs")

option(USE_LOCAL_RENDERER "Builds the project for pro-viz use" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 
cmake_policy(SET CMP0074 NEW)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(assimp REQUIRED)
find_package(ixwebsocket REQUIRED)
find_package(PNG REQUIRED)
find_package(glm REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
        CUDA_NVCC_FLAGS 
        ${CUDA_NVCC_FLAGS}; 
                -gencode arch=compute_70,code=sm_70
                -rdc=true
                -lineinfo
)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
cuda_add_executable(cuda_project ${SOURCES})

target_link_libraries(cuda_project assimp::assimp CUDA::nvml ixwebsocket::ixwebsocket PNG::PNG glm::glm)
target_link_libraries(cuda_project -lturbojpeg)
set_target_properties(cuda_project PROPERTIES CUDA_ARCHITECTURES "75,70")

if (USE_LOCAL_RENDERER)
        set(OpenGL_GL_PREFERENCE GLVND)

        find_package(glfw3 REQUIRED CONFIG)
        find_package(OpenGL REQUIRED)
        find_package(GLEW REQUIRED)

        target_link_libraries(cuda_project glfw OpenGL::GL GLEW::GLEW)
        target_compile_definitions(cuda_project PUBLIC USE_LOCAL_RENDERER)
endif()