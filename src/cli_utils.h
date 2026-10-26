#pragma once

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

// Host-only validation shared by the benchmark command-line interfaces.
namespace cli {

inline bool help_requested(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") return true;
    }
    return false;
}

inline bool positive_integer(const std::string& text, const char* name, long& value) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    // Several downstream cuBLAS/QR calls use 32-bit dimensions.
    if (text.empty() || end == text.c_str() || *end != '\0' || errno == ERANGE ||
        parsed <= 0 || parsed > INT_MAX) {
        std::fprintf(stderr, "Error: %s must be an integer in [1, %d], got '%s'.\n",
                     name, INT_MAX, text.c_str());
        return false;
    }
    value = parsed;
    return true;
}

template <typename T>
inline bool positive_real(const std::string& text, const char* name, T& value) {
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(text.c_str(), &end);
    if (text.empty() || end == text.c_str() || *end != '\0' || errno == ERANGE ||
        !std::isfinite(parsed) || parsed <= 0.0 ||
        parsed > static_cast<double>(std::numeric_limits<T>::max()) ||
        static_cast<T>(parsed) == T(0)) {
        std::fprintf(stderr, "Error: %s must be a finite positive number, got '%s'.\n",
                     name, text.c_str());
        return false;
    }
    value = static_cast<T>(parsed);
    return true;
}

inline bool boolean(const std::string& text, bool& value) {
    if (text == "on" || text == "true" || text == "1") value = true;
    else if (text == "off" || text == "false" || text == "0") value = false;
    else {
        std::fprintf(stderr, "Error: --padding requires on/off, got '%s'.\n", text.c_str());
        return false;
    }
    return true;
}

}  // namespace cli
