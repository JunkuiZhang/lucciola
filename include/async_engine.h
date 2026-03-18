#pragma once

#include "qwen.h"
#include "scheduler.h"
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lucciola {

struct RequestInfo {
    int max_new_tokens;
    int generated_tokens = 0;
    std::function<void(const std::string &next_token_text, bool is_finished)>
        stream_cb;
};

class AsyncEngine {
  public:
    AsyncEngine(
        const std::string &model_path, int max_blocks = 0, int block_size = 16);
    ~AsyncEngine();

    // Async thread-safe API to submit a generation request
    int generate_async(
        const std::string &prompt,
        int max_new_tokens,
        std::function<void(const std::string &, bool)> callback);

    // Stop execution
    void stop();

  private:
    void engine_loop();

    QwenModel model_;
    CacheConfig cache_config_;
    Scheduler scheduler_;

    std::unordered_map<int, RequestInfo> active_requests_;

    std::thread background_thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_flag_;

    int next_req_id_;

    // Pending queue: <req_id, prompt>
    std::vector<std::pair<int, std::string>> pending_prompts_;
};

} // namespace lucciola
