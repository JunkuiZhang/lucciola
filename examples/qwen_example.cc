#include "qwen.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << " Lucciola (Qwen3) Native CUDA Inference Core " << std::endl;
    std::cout << "=============================================" << std::endl;

    lucciola::QwenModel qwen("/home/zjk/cuda/lucciola/models/Qwen3-0.6B");

    std::string input = "To be or not to be, that's a question.";

    std::vector<int> input_ids = qwen.get_tokenizer().encode(input);

    std::vector<int> out_tokens = qwen.generate(input_ids, 50);

    std::cout << "\n\nOutput: ";
    for (int token : out_tokens) {
        std::cout << qwen.get_tokenizer().decode(token);
    }
    std::cout << "\n" << std::endl;

    std::cout << "Engine Shutting Down." << std::endl;
    return 0;
}