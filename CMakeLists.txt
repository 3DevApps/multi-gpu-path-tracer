cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp" "src/*.cu")
add_executable(cuda_project src/main.cu)