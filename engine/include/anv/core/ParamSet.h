// Node parameters.
//
// Everything is in ENGINEER UNITS — Hz, dB, ms, ratio — never normalized 0-1.
// Two reasons: the planning model writes dB and Hz natively, and a recipe
// stays reviewable and diffable by a human who is deciding whether to trust it.
//
// `items` carries ordered sub-parameter groups (an EQ's band list, for
// example) without needing a general-purpose JSON value type in the hot path.
#pragma once

#include <map>
#include <string>
#include <vector>

namespace anv {

class ParamSet {
public:
  ParamSet& set(const std::string& key, double value) {
    nums_[key] = value;
    return *this;
  }
  ParamSet& setStr(const std::string& key, std::string value) {
    strs_[key] = std::move(value);
    return *this;
  }
  ParamSet& addItem(ParamSet item) {
    items_.push_back(std::move(item));
    return *this;
  }

  double num(const std::string& key, double fallback) const {
    auto it = nums_.find(key);
    return it == nums_.end() ? fallback : it->second;
  }
  std::string str(const std::string& key, const std::string& fallback) const {
    auto it = strs_.find(key);
    return it == strs_.end() ? fallback : it->second;
  }
  bool has(const std::string& key) const {
    return nums_.count(key) != 0 || strs_.count(key) != 0;
  }

  // Numeric keys, so a test can sweep every parameter a node actually reads
  // without having to know its schema.
  std::vector<std::string> numericKeys() const {
    std::vector<std::string> keys;
    keys.reserve(nums_.size());
    for (const auto& kv : nums_) keys.push_back(kv.first);
    return keys;
  }

  const std::vector<ParamSet>& items() const { return items_; }
  std::vector<ParamSet>& items() { return items_; }

private:
  std::map<std::string, double> nums_;
  std::map<std::string, std::string> strs_;
  std::vector<ParamSet> items_;
};

} // namespace anv
