// Minimal argument parsing. Deliberately tiny -- the collector binaries take a
// handful of flags and pulling in a dependency for that would complicate the robot
// image for no benefit.
#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fpi {

class Args {
 public:
  Args(int argc, char** argv) {
    program_ = argc > 0 ? argv[0] : "fpi";
    for (int i = 1; i < argc; ++i) {
      std::string token(argv[i]);
      if (token.rfind("--", 0) != 0) {
        positional_.push_back(token);
        continue;
      }
      const auto eq = token.find('=');
      if (eq != std::string::npos) {
        options_[token.substr(2, eq - 2)] = token.substr(eq + 1);
      } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        options_[token.substr(2)] = argv[++i];
      } else {
        options_[token.substr(2)] = "true";
      }
    }
  }

  bool has(const std::string& key) const { return options_.count(key) > 0; }

  std::string get(const std::string& key, const std::string& fallback) const {
    const auto it = options_.find(key);
    return it == options_.end() ? fallback : it->second;
  }

  std::string require(const std::string& key) const {
    const auto it = options_.find(key);
    if (it == options_.end()) {
      throw std::runtime_error("missing required option --" + key);
    }
    return it->second;
  }

  double number(const std::string& key, double fallback) const {
    const auto it = options_.find(key);
    if (it == options_.end()) return fallback;
    return std::stod(it->second);
  }

  int integer(const std::string& key, int fallback) const {
    return static_cast<int>(number(key, fallback));
  }

  bool flag(const std::string& key) const {
    const auto it = options_.find(key);
    return it != options_.end() && it->second != "false" && it->second != "0";
  }

  const std::string& program() const { return program_; }

 private:
  std::string program_;
  std::map<std::string, std::string> options_;
  std::vector<std::string> positional_;
};

}  // namespace fpi
