#include <atomic>
#include <chrono>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <random>
#include <thread>
#include <vector>

using json = nlohmann::json;

std::atomic<bool> stop_flag{false};
std::atomic<int> total_requests{0};
std::atomic<int> total_tokens{0};
std::atomic<int> failed_requests{0};

void client_thread(int thread_id) {
    httplib::Client cli("localhost", 8080);
    cli.set_read_timeout(120, 0); // Allow long generation times

    std::vector<std::string> prompts = {
        "Explain the theory of relativity.",
        "Write a poem about a lost cat.",
        "What is the capital of France?",
        "Write a C++ hello world program.",
        "Tell me a short joke."};
    std::mt19937 rng(std::random_device{}() + thread_id);
    std::uniform_int_distribution<int> prompt_dist(0, prompts.size() - 1);
    std::uniform_int_distribution<int> token_dist(10, 50);

    while (!stop_flag) {
        json payload = {
            {"messages",
             {{{"role", "user"}, {"content", prompts[prompt_dist(rng)]}}}},
            {"max_tokens", token_dist(rng)}};

        auto res = cli.Post(
            "/v1/chat/completions", payload.dump(), "application/json");

        if (res && res->status == 200) {
            try {
                auto j_res = json::parse(res->body);
                int generated_tokens = 0;

                // Let's count some tokens depending on response size or parsed
                // count
                if (j_res.contains("usage") &&
                    j_res["usage"].contains("completion_tokens")) {
                    generated_tokens =
                        j_res["usage"]["completion_tokens"].get<int>();
                } else if (
                    j_res.contains("choices") && j_res["choices"].is_array() &&
                    !j_res["choices"].empty()) {
                    // Fallback approximation string size / 4
                    std::string text =
                        j_res["choices"][0]["message"]["content"];
                    generated_tokens = text.size() / 4;
                }

                total_tokens += std::max(1, generated_tokens);
                total_requests++;
            } catch (...) {
                failed_requests++;
            }
        } else {
            failed_requests++;
        }
    }
}

int main(int argc, char **argv) {
    int num_clients = 10;
    int duration_sec = 10;

    if (argc > 1)
        num_clients = std::stoi(argv[1]);
    if (argc > 2)
        duration_sec = std::stoi(argv[2]);

    std::cout << "[INFO] Starting benchmark with " << num_clients
              << " clients for " << duration_sec << " seconds.\n";

    auto start_time = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int i = 0; i < num_clients; ++i) {
        threads.emplace_back(client_thread, i);
    }

    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
    stop_flag = true;

    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double actual_duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end_time - start_time)
            .count();

    std::cout << "\n===== Benchmark Results =====\n";
    std::cout << "Duration:            " << actual_duration << " s\n";
    std::cout << "Total Requests:      " << total_requests << "\n";
    std::cout << "Failed Requests:     " << failed_requests << "\n";
    std::cout << "Total Tokens:        " << total_tokens << "\n";
    std::cout << "Throughput (Reqs/s): " << (total_requests / actual_duration)
              << "\n";
    std::cout << "Throughput (Toks/s): " << (total_tokens / actual_duration)
              << "\n";
    std::cout << "=============================\n";

    return 0;
}
