#include "async_engine.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>

int main() {
    std::cout << "=============================================\n";
    std::cout << "   Lucciola Async Continuous Batching Bench  \n";
    std::cout << "=============================================\n";

    std::string model_path = "/home/zjk/cuda/lucciola/models/Qwen3-0.6B";
    lucciola::AsyncEngine engine(model_path, 0, 16);

    int num_requests = 10;
    int max_tokens = 50;
    std::atomic<int> completed{0};

    std::string prompt = "Explain the architecture of a transformer model and "
                         "continuous batching in detail for benchmark.";

    auto start_time = std::chrono::high_resolution_clock::now();

    std::mutex mu;
    std::condition_variable cv;

    for (int i = 0; i < num_requests; i++) {
        engine.generate_async(
            prompt, max_tokens, [&](const std::string &text, bool is_finish) {
                if (is_finish) {
                    completed++;
                    std::cout << "\n[Req Done] Total completed: " << completed
                              << std::endl;
                    if (completed == num_requests) {
                        std::lock_guard<std::mutex> lock(mu);
                        cv.notify_one();
                    }
                }
            });
    }

    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&]() { return completed == num_requests; });

    auto end_time = std::chrono::high_resolution_clock::now();

    float total_time_s =
        std::chrono::duration<float>(end_time - start_time).count();
    float total_tokens = num_requests * max_tokens;
    float throughput = total_tokens / total_time_s;

    std::cout << "---------------------------------------------\n";
    std::cout << " Concurrency:       " << num_requests << " reqs\n";
    std::cout << " Output / req:      " << max_tokens << " tokens\n";
    std::cout << " Total Time:        " << total_time_s << " s\n";
    std::cout << " System Throughput: " << std::fixed << std::setprecision(2)
              << throughput << " tokens/sec\n";
    std::cout << "---------------------------------------------\n";

    engine.stop();
    return 0;
}
