#include "qwen.h"
#include <iostream>
#include <string>
#include <vector>

const int IM_START = 151644;
const int IM_END = 151645;
const int ENTER_ID = 1699;

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << " Lucciola (Qwen3) Native CUDA Inference Core " << std::endl;
    std::cout << "=============================================" << std::endl;

    // 1. Architecture Boot & Safetensors Loading & Tokenizer Config
    lucciola::QwenModel qwen("/home/zjk/cuda/lucciola/models/Qwen3-0.6B");

    std::cout
        << "Model warm. Entering interactive loop (type 'exit' to quit).\n"
        << std::endl;

    // 2. Continuous Interactive Run Loop
    while (true) {
        std::cout << ">> User: ";
        std::string input;
        if (!std::getline(std::cin, input) || input == "exit") {
            break;
        }

        // 2a. Real Tokenization
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

        std::cout << "<< Qwen: " << std::flush;

        // 2b. The Neural Network execution (Prefill generates 100 tokens)
        std::vector<int> out_tokens = qwen.generate(input_ids, 1000);

        // 2c. De-tokenization Output
        for (int token : out_tokens) {
            std::cout << qwen.get_tokenizer().decode(token) << std::flush;
        }
        std::cout << std::endl << std::endl;
    }

    std::cout << "Engine Shutting Down." << std::endl;
    return 0;
}