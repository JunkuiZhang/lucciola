#include "tokenizer.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <print>

using json = nlohmann::json;

namespace lucciola {

QwenTokenizer::QwenTokenizer(const std::string &vocab_path) {
    std::ifstream f(vocab_path);
    if (!f.is_open()) {
        std::println("Error: Failed to open tokenizer vocab: {}", vocab_path);
        return;
    }

    json vocab = json::parse(f);
    id_to_token_.resize(vocab.size());

    for (auto &el : vocab.items()) {
        std::string token = el.key();
        int id = el.value();
        token_to_id_[token] = id;
        if (id >= 0 && id < id_to_token_.size()) {
            id_to_token_[id] = token;
        }
    }
    std::println("Loaded {} tokens into Vocab.", token_to_id_.size());
}

std::vector<int> QwenTokenizer::encode(const std::string &text) const {
    std::vector<int> ids;
    std::string current = text;

    // Naive Greedy Maximal Prefix Match (approximates BPE without merge logic)
    while (!current.empty()) {
        int best_len = 0;
        int best_id = -1;

        // Find longest prefix that exists in vocab
        for (int len = current.size(); len > 0; --len) {
            std::string prefix = current.substr(0, len);
            auto it = token_to_id_.find(prefix);
            if (it != token_to_id_.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }

        if (best_len > 0) {
            ids.push_back(best_id);
            current = current.substr(best_len);
        } else {
            // Fallback: push byte individually if mapped, else ignore
            std::string single_char = current.substr(0, 1);
            auto it = token_to_id_.find(single_char);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            }
            current = current.substr(1);
        }
    }
    return ids;
}

std::string QwenTokenizer::decode(int id) const {
    if (id >= 0 && id < (int)id_to_token_.size()) {
        // A naive decode. Real tiktoken BPE decodes bytes to utf-8.
        // We will just return the literal mapping string.
        return id_to_token_[id];
    }
    return "";
}

} // namespace lucciola
