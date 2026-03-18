#include "qwen.h"
#include <iostream>
#include <string>
#include <vector>

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
        std::vector<int> input_ids = qwen.get_tokenizer().encode(input);

        std::cout << "<< Qwen: " << std::flush;

        // 2b. The Neural Network execution (Prefill generates 100 tokens)
        std::vector<int> out_tokens = qwen.generate(input_ids, 100);

        // 2c. De-tokenization Output
        for (int token : out_tokens) {
            std::cout << qwen.get_tokenizer().decode(token) << std::flush;
        }
        std::cout << std::endl << std::endl;
    }

    std::cout << "Engine Shutting Down." << std::endl;
    return 0;
}