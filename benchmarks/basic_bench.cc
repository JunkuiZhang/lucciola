#include "qwen.h"
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "        Lucciola Performance Benchmark       " << std::endl;
    std::cout << "=============================================" << std::endl;

    // Load Model
    lucciola::QwenModel qwen("/home/zjk/cuda/lucciola/models/Qwen3-0.6B");

    // Prepare Prompt
    std::string prompt = "Benchmark the Time-to-First-Token and Decode latency by running this long prompt through the model multiple times to measure average tokens per second perfectly. The primary object is high GPU utilization.";
    std::vector<int> input_ids = qwen.get_tokenizer().encode(prompt);
    
    std::cout << "Prompt Length: " << input_ids.size() << " tokens" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup (to avoid initial context load latency)
    int next_token = qwen.prefill(input_ids);

    // Benchmarking TTFT
    cudaEventRecord(start);
    int first_token = qwen.prefill(input_ids);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ttft_ms = 0;
    cudaEventElapsedTime(&ttft_ms, start, stop);
    
    // Benchmarking Decode
    int decode_steps = 100;
    int current_seq_len = input_ids.size();
    
    cudaEventRecord(start);
    for (int step = 0; step < decode_steps; ++step) {
        first_token = qwen.decode(first_token, current_seq_len++);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float decode_total_ms = 0;
    cudaEventElapsedTime(&decode_total_ms, start, stop);
    
    float avg_decode_ms = decode_total_ms / decode_steps;
    float tokens_per_sec = 1000.0f / avg_decode_ms;
    
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << " TTFT (Prompt=" << input_ids.size() << "t): " << ttft_ms << " ms" << std::endl;
    std::cout << " Decode Latency: " << avg_decode_ms << " ms/token" << std::endl;
    std::cout << " Decode Speed:   " << tokens_per_sec << " tokens/sec" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
