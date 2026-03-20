#include "block_manager.h"
#include "qwen.h"
#include "scheduler.h"
#include "sequence.h"
#include <iostream>
#include <string>
#include <vector>

const int IM_START = 151644;
const int IM_END = 151645;
const int ENTER_ID = 1699;

int main() {
    std::cout << "========================================================"
              << std::endl;
    std::cout << " Lucciola Continuous Batching (vLLM PagedAttention Core) "
              << std::endl;
    std::cout << "========================================================"
              << std::endl;

    // 1. Architecture Boot & Safetensors Loading & Tokenizer Config
    lucciola::QwenModel qwen("/home/zjk/cuda/lucciola/models/Qwen3-0.6B");

    // Initialize Paged KV Cache memory pool
    int max_blocks = 1024;
    int block_size = 16;
    qwen.init_paged_kv_cache(max_blocks, block_size);

    lucciola::CacheConfig cache_config = {block_size, max_blocks, 0};
    lucciola::Scheduler scheduler(cache_config);

    std::cout
        << "Model warm (Paged Cache Engine Start). Entering interactive loop."
        << std::endl;

    int request_id = 0;

    // We will just demonstrate one request for now but using the dynamic
    // scheduler!
    while (true) {
        std::cout << ">> User: ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit") {
            break;
        }

        request_id++;

        std::vector<int> input_ids;
        input_ids.push_back(IM_START);
        auto sys_ids =
            qwen.get_tokenizer().encode("system\nYou are a helpful assistant.");
        input_ids.insert(input_ids.end(), sys_ids.begin(), sys_ids.end());
        input_ids.push_back(IM_END);
        input_ids.push_back(ENTER_ID);

        input_ids.push_back(IM_START);
        auto user_ids = qwen.get_tokenizer().encode("user\n" + input);
        input_ids.insert(input_ids.end(), user_ids.begin(), user_ids.end());
        input_ids.push_back(IM_END);
        input_ids.push_back(ENTER_ID);

        input_ids.push_back(IM_START);
        auto assistant_ids = qwen.get_tokenizer().encode("assistant\n");
        input_ids.insert(
            input_ids.end(), assistant_ids.begin(), assistant_ids.end());

        // Create a Sequence
        auto seq = std::make_shared<lucciola::Sequence>(
            request_id, input_ids, block_size);
        scheduler.add_request(seq);

        std::cout << "<< Qwen: " << std::flush;

        // Continuous Batching Loop
        int generated_count = 0;

        while (scheduler.has_unfinished_requests()) {
            lucciola::InputMetadata meta = scheduler.step();

            if (meta.num_seqs > 0) {
                // Forward pass using PagedAttention
                qwen.step_with_paged_attention(meta);

                // For simplicity, handle just the 1 sequence for printing
                // output
                int next_token = meta.seqs[0]->get_token_ids().back();

                // Print only in decode phase
                if (0 < meta.num_decode_seqs) {
                    std::cout << qwen.get_tokenizer().decode(next_token)
                              << std::flush;
                    generated_count++;
                }

                // Check termination (simplified for 1 seq)
                if (next_token == 151645 || next_token == 151643 ||
                    generated_count >= 100) {
                    scheduler.free_sequence(meta.seqs[0]);
                }
            } else {
                // Wait queue empty or blocks exhausted
                break;
            }
        }
        std::cout << std::endl << std::endl;
    }

    std::cout << "Engine Shutting Down." << std::endl;
    return 0;
}
