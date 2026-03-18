#include "async_engine.h"
#include <condition_variable>
#include <httplib.h>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <sstream>
#include <thread>

using json = nlohmann::json;
using namespace lucciola;

int main(int argc, char **argv) {
    std::string model_path = "/home/zjk/cuda/lucciola/models/Qwen3-0.6B";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "[INFO] Loading AsyncEngine with model: " << model_path
              << "\n";
    AsyncEngine engine(model_path, 1024, 16);

    httplib::Server svr;

    svr.Post(
        "/v1/chat/completions",
        [&](const httplib::Request &req, httplib::Response &res) {
            try {
                auto j_req = json::parse(req.body);
                std::string prompt = "";

                // Basic extraction of messages (simplified)
                if (j_req.contains("messages") &&
                    j_req["messages"].is_array()) {
                    for (const auto &msg : j_req["messages"]) {
                        if (msg.contains("role") && msg.contains("content")) {
                            // Assuming system/user/assistant formatting is
                            // handled by AsyncEngine Let's just extract the
                            // last user message for this simple mock
                            if (msg["role"] == "user") {
                                prompt = msg["content"].get<std::string>();
                            }
                        }
                    }
                } else if (j_req.contains("prompt")) {
                    prompt = j_req["prompt"].get<std::string>();
                }

                bool stream = j_req.value("stream", false);
                int max_tokens = j_req.value("max_tokens", 512);

                if (prompt.empty()) {
                    res.status = 400;
                    res.set_content(
                        R"({"error": "Prompt or messages missing"})",
                        "application/json");
                    return;
                }

                if (stream) {
                    res.set_chunked_content_provider(
                        "text/event-stream",
                        [&engine, prompt, max_tokens](
                            size_t offset, httplib::DataSink &sink) {
                            std::mutex mu;
                            std::condition_variable cv;
                            bool done = false;
                            std::queue<std::string> chunks;

                            engine.generate_async(
                                prompt,
                                max_tokens,
                                [&](const std::string &text, bool is_finished) {
                                    std::lock_guard<std::mutex> lock(mu);
                                    // OpenAI format chunk
                                    json j_chunk = {
                                        {"choices",
                                         {{{"delta", {{"content", text}}},
                                           {"finish_reason",
                                            is_finished ? "stop" : nullptr}}}}};
                                    std::string ssev =
                                        "data: " + j_chunk.dump() + "\n\n";
                                    if (is_finished) {
                                        ssev += "data: [DONE]\n\n";
                                    }
                                    chunks.push(ssev);
                                    done = done || is_finished;
                                    cv.notify_one();
                                });

                            while (true) {
                                std::unique_lock<std::mutex> lock(mu);
                                cv.wait(lock, [&] {
                                    return !chunks.empty() || done;
                                });
                                while (!chunks.empty()) {
                                    std::string c = chunks.front();
                                    chunks.pop();
                                    sink.write(c.c_str(), c.size());
                                }
                                if (done && chunks.empty()) {
                                    sink.done();
                                    return false; // stop provider
                                }
                            }
                            return true;
                        });
                } else {
                    // Non-streaming
                    std::mutex mu;
                    std::condition_variable cv;
                    bool done = false;
                    std::string full_response = "";

                    engine.generate_async(
                        prompt,
                        max_tokens,
                        [&](const std::string &text, bool is_finished) {
                            std::lock_guard<std::mutex> lock(mu);
                            full_response += text;
                            if (is_finished)
                                done = true;
                            cv.notify_one();
                        });

                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return done; });

                    json j_res = {
                        {"choices",
                         {{{"message",
                            {{"role", "assistant"},
                             {"content", full_response}}},
                           {"finish_reason", "stop"}}}}};
                    res.set_content(j_res.dump(), "application/json");
                }
            } catch (const std::exception &e) {
                res.status = 500;
                json j_err = {{"error", e.what()}};
                res.set_content(j_err.dump(), "application/json");
            }
        });

    std::cout << "[INFO] HTTP Server listening on 0.0.0.0:8080...\n";
    svr.listen("0.0.0.0", 8080);

    engine.stop();
    return 0;
}
