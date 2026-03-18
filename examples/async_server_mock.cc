#include "async_engine.h"
#include <iostream>
#include <string>
#include <vector>

using namespace lucciola;

int main() {
    std::string model_path =
        "/home/zjk/cuda/lucciola/models/Qwen3-0.6B"; // Update path if needed

    std::cout << "[INFO] Loading AsyncEngine with model: " << model_path
              << "\n";
    AsyncEngine engine(model_path, 1024, 16); // 1024 blocks, block_size 16

    std::cout << "[INFO] Submitting mock requests...\n";
    std::vector<std::string> prompts = {
        "what is 1+1", "describe the color of the sky", "hello!"};

    for (const auto &p : prompts) {
        engine.generate_async(
            p, 50, [&p](const std::string &text, bool is_finished) {
                std::cout << "<Req " << p << "> " << text
                          << (is_finished ? " [DONE]\n" : "");
                std::cout.flush();
            });
    }

    std::cout
        << "[INFO] Sleeping for 10 seconds to allow async processing...\n";
    std::this_thread::sleep_for(std::chrono::seconds(10));

    std::cout << "[INFO] Shutting down...\n";
    engine.stop();
    return 0;
}
