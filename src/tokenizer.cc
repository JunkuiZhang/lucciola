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

// Build the forward GPT-2 bytes_to_unicode table.
// Maps each raw byte to a UTF-8 string representing its vocab codepoint.
static std::vector<std::string> build_byte_to_unicode_utf8() {
    // First determine which bytes map to themselves
    std::unordered_map<int, char32_t> b2u;
    auto add_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b)
            b2u[b] = static_cast<char32_t>(b);
    };
    add_range(0x21, 0x7E);   // '!' .. '~'
    add_range(0xA1, 0xAC);   // '¡' .. '¬'
    add_range(0xAE, 0xFF);   // '®' .. 'ÿ'

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (b2u.find(b) == b2u.end()) {
            b2u[b] = static_cast<char32_t>(256 + n);
            ++n;
        }
    }

    // Convert each codepoint to its UTF-8 string
    auto cp_to_utf8 = [](char32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        return s;
    };

    std::vector<std::string> table(256);
    for (int b = 0; b < 256; ++b) {
        table[b] = cp_to_utf8(b2u[b]);
    }
    return table;
}

std::vector<int> QwenTokenizer::encode(const std::string &text) const {
    static const auto byte_to_unicode = build_byte_to_unicode_utf8();

    // Step 1: Convert raw bytes to their GPT-2 unicode representation
    std::string mapped;
    mapped.reserve(text.size() * 2);
    for (unsigned char c : text) {
        mapped.append(byte_to_unicode[c]);
    }

    // Step 2: Greedy longest-prefix match on the mapped string
    std::vector<int> ids;
    std::string current = mapped;

    while (!current.empty()) {
        int best_len = 0;
        int best_id = -1;

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
            // Skip one UTF-8 character if nothing matches
            uint8_t c = static_cast<uint8_t>(current[0]);
            int skip = 1;
            if ((c >> 5) == 0x06) skip = 2;
            else if ((c >> 4) == 0x0E) skip = 3;
            else if ((c >> 3) == 0x1E) skip = 4;
            current = current.substr(skip);
        }
    }
    return ids;
}

// Build the reverse of GPT-2's bytes_to_unicode table.
// In the vocab, certain bytes are represented as Unicode codepoints
// (e.g. space 0x20 -> Ġ U+0120). This table maps them back.
static std::unordered_map<char32_t, uint8_t> build_unicode_to_byte() {
    // First, collect the "direct" codepoints (printable ASCII + Latin-1 ranges)
    std::unordered_map<char32_t, uint8_t> u2b;
    auto add_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b)
            u2b[static_cast<char32_t>(b)] = static_cast<uint8_t>(b);
    };
    add_range(0x21, 0x7E);   // '!' .. '~'
    add_range(0xA1, 0xAC);   // '¡' .. '¬'
    add_range(0xAE, 0xFF);   // '®' .. 'ÿ'

    // Remaining bytes (0x00-0x20, 0x7F-0xA0, 0xAD) are mapped to U+0100..
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (u2b.find(static_cast<char32_t>(b)) == u2b.end()) {
            u2b[static_cast<char32_t>(256 + n)] = static_cast<uint8_t>(b);
            ++n;
        }
    }
    return u2b;
}

std::string QwenTokenizer::decode(int id) const {
    if (id < 0 || id >= (int)id_to_token_.size()) {
        return "";
    }

    static const auto unicode_to_byte = build_unicode_to_byte();

    const std::string &token = id_to_token_[id];
    std::string result;
    result.reserve(token.size());

    // Parse each UTF-8 codepoint and map it back to the original byte
    size_t i = 0;
    while (i < token.size()) {
        char32_t cp = 0;
        uint8_t c = static_cast<uint8_t>(token[i]);

        int len = 1;
        if (c < 0x80) {
            cp = c;
        } else if ((c >> 5) == 0x06) {     // 110xxxxx -> 2-byte
            cp = c & 0x1F;
            len = 2;
        } else if ((c >> 4) == 0x0E) {     // 1110xxxx -> 3-byte
            cp = c & 0x0F;
            len = 3;
        } else if ((c >> 3) == 0x1E) {     // 11110xxx -> 4-byte
            cp = c & 0x07;
            len = 4;
        }

        for (int j = 1; j < len && (i + j) < token.size(); ++j) {
            cp = (cp << 6) | (static_cast<uint8_t>(token[i + j]) & 0x3F);
        }
        i += len;

        auto it = unicode_to_byte.find(cp);
        if (it != unicode_to_byte.end()) {
            result.push_back(static_cast<char>(it->second));
        } else {
            // Codepoint not in the mapping — emit original UTF-8 bytes
            result.append(token, i - len, len);
        }
    }
    return result;
}

} // namespace lucciola
