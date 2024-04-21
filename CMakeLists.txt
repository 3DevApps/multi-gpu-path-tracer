cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(
        CUDA_NVCC_FLAGS 
        ${CUDA_NVCC_FLAGS}; 
                -gencode arch=compute_75,code=sm_75 -rdc=true
)

cuda_add_executable(
        cuda_project 
        src/main.cu
)
set_target_properties(cuda_project PROPERTIES CUDA_ARCHITECTURES "75")
