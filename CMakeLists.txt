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

file(GLOB SOURCES "src/*.cpp" "src/*.cu")

add_executable(cuda_project src/main.cu)

target_link_libraries(cuda_project PRIVATE glfw OpenGL::GL GLEW::GLEW assimp::assimp)
