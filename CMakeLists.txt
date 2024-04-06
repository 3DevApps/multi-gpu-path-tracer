cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(PROJECT_NAME "cuda_project")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

execute_process(COMMAND bash scripts/install_dependencies.sh)

find_package(assimp REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp" "src/*.cu" "tests/*.cpp")

add_executable(${PROJECT_NAME} src/main.cu)

target_include_directories(${PROJECT_NAME}
                           PUBLIC
                           ${ASSIMP_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
                      ${ASSIMP_LIBRARIES})

                      