#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace lucciola {

class QwenTokenizer {
  public:
    QwenTokenizer(const std::string &vocab_path);

    std::vector<int> encode(const std::string &text) const;
    std::string decode(int id) const;

  private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
};

} // namespace lucciola
