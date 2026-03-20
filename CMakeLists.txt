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
  httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

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
    src/tokenizer.cc
    src/block_manager.cc
    src/sequence.cc
    src/scheduler.cc
    src/async_engine.cc
    src/safetensors.cc
    src/gpu_arena.cc
    src/kernels/embedding.cu
    src/kernels/rmsnorm.cu
    src/kernels/linear.cu
    src/kernels/rope.cu
    src/kernels/attention.cu
    src/kernels/swiglu.cu
    src/kernels/argmax.cu
    src/kernels/attention_paged.cu
    src/kernels/add.cu
    src/kernels/learn/softmax.cu
)
target_link_libraries(lucciola PUBLIC nlohmann_json::nlohmann_json CUDA::cudart)

add_executable(lucciola_chat examples/qwen_dynamic.cc)
target_link_libraries(lucciola_chat PRIVATE lucciola)

add_executable(lucciola_example examples/qwen_example.cc)
target_link_libraries(lucciola_example PRIVATE lucciola)
add_executable(async_server_mock examples/async_server_mock.cc)
target_link_libraries(async_server_mock PRIVATE lucciola)

# benchmarks
add_executable(basic_bench benchmarks/basic_bench.cc)
target_link_libraries(basic_bench PRIVATE lucciola)

add_executable(http_server examples/http_server.cc)
target_link_libraries(http_server PRIVATE lucciola httplib nlohmann_json::nlohmann_json)

add_executable(async_bench benchmarks/async_bench.cc)
target_link_libraries(async_bench PRIVATE lucciola)

add_executable(benchmark_server benchmarks/benchmark_server.cc)
target_link_libraries(benchmark_server PRIVATE lucciola httplib nlohmann_json::nlohmann_json)

add_executable(benchmark_client benchmarks/benchmark_client.cc)
target_link_libraries(benchmark_client PRIVATE httplib nlohmann_json::nlohmann_json)

# learn
add_executable(learn_softmax learn/softmax/main.cc)
target_link_libraries(learn_softmax PRIVATE lucciola)
