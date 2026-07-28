// A small, strict JSON reader.
//
// WHY NOT A LIBRARY. nlohmann/json is MIT and excellent and would have been a
// perfectly defensible choice. It is also thirty thousand lines compiled into
// every translation unit that touches a recipe, for a job that is three
// hundred. The deciding factor is not size though — it is that this parser is
// the boundary where a language model's output becomes something that
// allocates memory and configures DSP, and that boundary is worth being able
// to read end to end.
//
// STRICT ON PURPOSE. No trailing commas, no comments, no NaN or Infinity
// literals, no single quotes, no unquoted keys. A recipe is machine-generated
// and machine-consumed; every leniency is a way for two implementations to
// disagree about what a file means, and a recipe that means different things
// in different places is a reproducibility bug rather than a convenience.
//
// DEPTH IS BOUNDED. A recursive-descent parser on unbounded input is a stack
// overflow waiting to happen, and `[[[[[[...` is four bytes of attacker effort
// per level. The limit is generous next to any real recipe and fatal to that
// attack.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace anv {

class JsonError : public std::runtime_error {
public:
  explicit JsonError(const std::string& what) : std::runtime_error(what) {}
};

class JsonValue {
public:
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;

  bool isNull() const { return type == Type::Null; }
  bool isNumber() const { return type == Type::Number; }
  bool isString() const { return type == Type::String; }
  bool isBool() const { return type == Type::Bool; }
  bool isArray() const { return type == Type::Array; }
  bool isObject() const { return type == Type::Object; }

  // Lookup returning null rather than throwing, so callers can express
  // "optional with a default" without a has()/get() pair that reads the map
  // twice and can disagree with itself.
  const JsonValue* find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }

  double numberOr(const std::string& key, double fallback) const {
    const JsonValue* v = find(key);
    return (v && v->isNumber()) ? v->number : fallback;
  }
  std::string stringOr(const std::string& key, const std::string& fallback) const {
    const JsonValue* v = find(key);
    return (v && v->isString()) ? v->string : fallback;
  }
  bool boolOr(const std::string& key, bool fallback) const {
    const JsonValue* v = find(key);
    return (v && v->isBool()) ? v->boolean : fallback;
  }
};

namespace detail {

class JsonParser {
public:
  explicit JsonParser(const std::string& text) : s_(text) {}

  JsonValue parse() {
    skipWhitespace();
    JsonValue v = parseValue(0);
    skipWhitespace();
    if (p_ != s_.size())
      fail("trailing content after the top-level value");
    return v;
  }

private:
  static constexpr int kMaxDepth = 64;

  [[noreturn]] void fail(const std::string& what) const {
    // Byte offset, not line/column: recipes are machine-written and usually
    // arrive on one line, where a column number is the only useful coordinate
    // and happens to equal the offset.
    throw JsonError("JSON at byte " + std::to_string(p_) + ": " + what);
  }

  void skipWhitespace() {
    while (p_ < s_.size()) {
      const char c = s_[p_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++p_;
      else break;
    }
  }

  char peek() const {
    if (p_ >= s_.size()) fail("unexpected end of input");
    return s_[p_];
  }

  void expect(char c) {
    if (p_ >= s_.size() || s_[p_] != c)
      fail(std::string("expected '") + c + "'");
    ++p_;
  }

  bool literal(const char* word) {
    const size_t n = std::char_traits<char>::length(word);
    if (s_.compare(p_, n, word) != 0) return false;
    p_ += n;
    return true;
  }

  JsonValue parseValue(int depth) {
    if (depth > kMaxDepth) fail("nesting deeper than 64 levels");
    switch (peek()) {
      case '{': return parseObject(depth);
      case '[': return parseArray(depth);
      case '"': {
        JsonValue v;
        v.type = JsonValue::Type::String;
        v.string = parseString();
        return v;
      }
      case 't': {
        if (!literal("true")) fail("invalid literal");
        JsonValue v; v.type = JsonValue::Type::Bool; v.boolean = true; return v;
      }
      case 'f': {
        if (!literal("false")) fail("invalid literal");
        JsonValue v; v.type = JsonValue::Type::Bool; v.boolean = false; return v;
      }
      case 'n': {
        if (!literal("null")) fail("invalid literal");
        return JsonValue{};
      }
      default: return parseNumber();
    }
  }

  JsonValue parseObject(int depth) {
    expect('{');
    JsonValue v;
    v.type = JsonValue::Type::Object;

    skipWhitespace();
    if (peek() == '}') { ++p_; return v; }

    for (;;) {
      skipWhitespace();
      if (peek() != '"') fail("object keys must be quoted strings");
      const std::string key = parseString();
      // A duplicate key is not a merge and not a last-one-wins: it is two
      // different documents depending on who reads it. Reject it.
      if (v.object.count(key)) fail("duplicate object key \"" + key + "\"");

      skipWhitespace();
      expect(':');
      skipWhitespace();
      v.object.emplace(key, parseValue(depth + 1));

      skipWhitespace();
      const char c = peek();
      if (c == ',') { ++p_; continue; }
      if (c == '}') { ++p_; return v; }
      fail("expected ',' or '}' in object");
    }
  }

  JsonValue parseArray(int depth) {
    expect('[');
    JsonValue v;
    v.type = JsonValue::Type::Array;

    skipWhitespace();
    if (peek() == ']') { ++p_; return v; }

    for (;;) {
      skipWhitespace();
      v.array.push_back(parseValue(depth + 1));
      skipWhitespace();
      const char c = peek();
      if (c == ',') { ++p_; continue; }
      if (c == ']') { ++p_; return v; }
      fail("expected ',' or ']' in array");
    }
  }

  std::string parseString() {
    expect('"');
    std::string out;
    for (;;) {
      if (p_ >= s_.size()) fail("unterminated string");
      const unsigned char c = static_cast<unsigned char>(s_[p_++]);

      if (c == '"') return out;
      if (c < 0x20) fail("raw control character in string");
      if (c != '\\') { out.push_back(static_cast<char>(c)); continue; }

      if (p_ >= s_.size()) fail("unterminated escape");
      const char e = s_[p_++];
      switch (e) {
        case '"':  out.push_back('"');  break;
        case '\\': out.push_back('\\'); break;
        case '/':  out.push_back('/');  break;
        case 'b':  out.push_back('\b'); break;
        case 'f':  out.push_back('\f'); break;
        case 'n':  out.push_back('\n'); break;
        case 'r':  out.push_back('\r'); break;
        case 't':  out.push_back('\t'); break;
        case 'u':  appendUtf8(parseHex4(), out); break;
        default:   fail("unknown escape sequence");
      }
    }
  }

  uint32_t parseHex4() {
    if (p_ + 4 > s_.size()) fail("truncated \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[p_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else fail("non-hex digit in \\u escape");
    }
    return v;
  }

  void appendUtf8(uint32_t cp, std::string& out) {
    // Surrogate pair. A lone high surrogate followed by anything else is
    // malformed; emitting the replacement character instead of failing would
    // silently change a preset name or an artist alias, which is exactly the
    // kind of quiet corruption a strict parser exists to prevent.
    if (cp >= 0xD800 && cp <= 0xDBFF) {
      if (p_ + 6 > s_.size() || s_[p_] != '\\' || s_[p_ + 1] != 'u')
        fail("unpaired high surrogate");
      p_ += 2;
      const uint32_t lo = parseHex4();
      if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate");
      cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
      fail("unpaired low surrogate");
    }

    auto put = [&out](uint32_t byte) {
      out.push_back(static_cast<char>(byte & 0xFF));
    };
    if (cp < 0x80) {
      put(cp);
    } else if (cp < 0x800) {
      put(0xC0 | (cp >> 6));
      put(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      put(0xE0 | (cp >> 12));
      put(0x80 | ((cp >> 6) & 0x3F));
      put(0x80 | (cp & 0x3F));
    } else {
      put(0xF0 | (cp >> 18));
      put(0x80 | ((cp >> 12) & 0x3F));
      put(0x80 | ((cp >> 6) & 0x3F));
      put(0x80 | (cp & 0x3F));
    }
  }

  JsonValue parseNumber() {
    const size_t start = p_;
    if (p_ < s_.size() && s_[p_] == '-') ++p_;

    // JSON forbids leading zeros: "01" is not a number. Accepting it would
    // make a recipe that round-trips through a stricter parser change meaning.
    if (p_ < s_.size() && s_[p_] == '0') {
      ++p_;
    } else {
      if (p_ >= s_.size() || !isDigit(s_[p_])) fail("expected a number");
      while (p_ < s_.size() && isDigit(s_[p_])) ++p_;
    }

    if (p_ < s_.size() && s_[p_] == '.') {
      ++p_;
      if (p_ >= s_.size() || !isDigit(s_[p_])) fail("digits required after '.'");
      while (p_ < s_.size() && isDigit(s_[p_])) ++p_;
    }

    if (p_ < s_.size() && (s_[p_] == 'e' || s_[p_] == 'E')) {
      ++p_;
      if (p_ < s_.size() && (s_[p_] == '+' || s_[p_] == '-')) ++p_;
      if (p_ >= s_.size() || !isDigit(s_[p_])) fail("digits required in exponent");
      while (p_ < s_.size() && isDigit(s_[p_])) ++p_;
    }

    JsonValue v;
    v.type = JsonValue::Type::Number;
    // std::stod would THROW std::out_of_range on 1e400 — a different exception
    // type escaping this parser, so a caller catching JsonError would miss it
    // and the process would abort on an input a user can type. strtod reports
    // the same condition as a return value, which the finiteness check below
    // converts into a JsonError like every other malformed input.
    const std::string text = s_.substr(start, p_ - start);
    v.number = std::strtod(text.c_str(), nullptr);
    if (!std::isfinite(v.number))
      fail("number " + text + " is not finite");
    return v;
  }

  static bool isDigit(char c) { return c >= '0' && c <= '9'; }

  const std::string& s_;
  size_t p_ = 0;
};

} // namespace detail

inline JsonValue parseJson(const std::string& text) {
  return detail::JsonParser(text).parse();
}

} // namespace anv
