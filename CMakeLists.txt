cmake_minimum_required(VERSION 3.35)

project(lucciola VERSION 0.1.0 LANGUAGES CXX CUDA)

include_directories(include)
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# CUDA Settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

# Cpp Settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lucciola STATIC
    src/qwen.cc
    src/safetensors.cc
    src/gpu_arena.cc
    src/kernels/embedding.cu
    src/kernels/rmsnorm.cu
    src/kernels/linear.cu
    src/kernels/rope.cu
)
target_link_libraries(lucciola PUBLIC nlohmann_json::nlohmann_json CUDA::cudart)

add_executable(lucciola_chat examples/qwen.cc)
target_link_libraries(lucciola_chat PRIVATE lucciola)
