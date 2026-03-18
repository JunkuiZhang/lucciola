#include "async_engine.h"
#include <condition_variable>
#include <httplib.h>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace lucciola;

int main(int argc, char **argv) {
    std::string model_path =
        "/home/zjk/cuda/lucciola/models/Qwen3-0.6B"; // Typical bench target
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "[INFO] Loading AsyncEngine with model: " << model_path
              << "\n";
    // We pass 0 for max_blocks to enable the vLLM style 90% auto-allocation you
    // requested.
    AsyncEngine engine(model_path, 0, 16);

    httplib::Server svr;

    svr.Post(
        "/v1/chat/completions",
        [&](const httplib::Request &req, httplib::Response &res) {
            try {
                auto j_req = json::parse(req.body);
                std::string prompt = "benchmark default prompt";

                if (j_req.contains("messages") &&
                    j_req["messages"].is_array()) {
                    for (const auto &msg : j_req["messages"]) {
                        if (msg.contains("role") && msg.contains("content") &&
                            msg["role"] == "user") {
                            prompt = msg["content"].get<std::string>();
                        }
                    }
                } else if (j_req.contains("prompt")) {
                    prompt = j_req["prompt"].get<std::string>();
                }

                int max_tokens = j_req.value("max_tokens", 128);

                // Using Non-streaming for benchmark simplicity by default
                std::mutex mu;
                std::condition_variable cv;
                bool done = false;
                std::string full_response = "";
                int generated = 0;

                engine.generate_async(
                    prompt,
                    max_tokens,
                    [&](const std::string &text, bool is_finished) {
                        std::lock_guard<std::mutex> lock(mu);
                        full_response += text;
                        generated++;
                        if (is_finished) {
                            done = true;
                        }
                        cv.notify_one();
                    });

                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return done; });

                json j_res = {
                    {"choices",
                     {{{"message",
                        {{"role", "assistant"}, {"content", full_response}}},
                       {"finish_reason", "stop"}}}},
                    {"usage", {{"completion_tokens", generated}}}};
                res.set_content(j_res.dump(), "application/json");

            } catch (const std::exception &e) {
                res.status = 500;
                json j_err = {{"error", e.what()}};
                res.set_content(j_err.dump(), "application/json");
            }
        });

    std::cout << "[INFO] Benchmark HTTP Server listening on 0.0.0.0:8080...\n";
    svr.listen("0.0.0.0", 8080);

    engine.stop();
    return 0;
}
