// Minimal self-registering test harness (no external framework).
#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

namespace vtt::test {

struct TestCase {
    const char* name;
    void (*function)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

/// Number of failed checks so far (mutated by the CHECK macros).
inline int failures = 0;

struct Registrar {
    Registrar(const char* name, void (*function)()) { registry().push_back({name, function}); }
};

inline void report_failure(const char* file, int line, const char* expression) {
    ++failures;
    std::cerr << "  FAILED " << file << ':' << line << ": " << expression << '\n';
}

inline void report_near(const char* file, int line, const char* expression, double a, double b,
                        double tolerance) {
    ++failures;
    std::cerr << "  FAILED " << file << ':' << line << ": " << expression << " (" << a << " vs " << b
              << ", tol " << tolerance << ")\n";
}

/// Run all registered tests (or only those whose name contains argv[1]).
inline int run_all(int argc, char** argv) {
    const std::string_view filter = argc > 1 ? std::string_view(argv[1]) : std::string_view();
    int executed = 0;
    for (const TestCase& test : registry()) {
        if (!filter.empty() && std::string_view(test.name).find(filter) == std::string_view::npos) {
            continue;
        }
        const int before = failures;
        test.function();
        ++executed;
        std::cout << (failures == before ? "[ OK ] " : "[FAIL] ") << test.name << '\n';
    }
    std::cout << executed << " test(s), " << failures << " failure(s)\n";
    return failures == 0 && executed > 0 ? 0 : 1;
}

}  // namespace vtt::test

#define VTT_TEST(name)                                                     \
    static void name();                                                    \
    static const ::vtt::test::Registrar name##_registrar{#name, &name};    \
    static void name()

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (static_cast<bool>(condition)) {                                   \
        } else {                                                              \
            ::vtt::test::report_failure(__FILE__, __LINE__, #condition);      \
        }                                                                     \
    } while (false)

#define CHECK_NEAR(a, b, tolerance)                                                       \
    do {                                                                                  \
        const double vtt_a_ = static_cast<double>(a);                                     \
        const double vtt_b_ = static_cast<double>(b);                                     \
        const double vtt_t_ = static_cast<double>(tolerance);                             \
        if (!(std::abs(vtt_a_ - vtt_b_) <= vtt_t_)) {                                     \
            ::vtt::test::report_near(__FILE__, __LINE__, #a " ~ " #b, vtt_a_, vtt_b_, vtt_t_); \
        }                                                                                 \
    } while (false)

#define VTT_TEST_MAIN() \
    int main(int argc, char** argv) { return ::vtt::test::run_all(argc, argv); }
