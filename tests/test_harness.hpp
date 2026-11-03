// Minimal self-registering test harness (no external framework).
#pragma once

#include <cmath>
#include <cstdio>
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

inline int& failure_count() {
    static int failures = 0;
    return failures;
}

struct Registrar {
    Registrar(const char* name, void (*function)()) { registry().push_back({name, function}); }
};

inline void report_failure(const char* file, int line, const char* expression) {
    ++failure_count();
    std::fprintf(stderr, "  FAILED %s:%d: %s\n", file, line, expression);
}

inline void report_near(const char* file, int line, const char* expression, double a, double b,
                        double tolerance) {
    ++failure_count();
    std::fprintf(stderr, "  FAILED %s:%d: %s (%.6g vs %.6g, tol %.3g)\n", file, line, expression, a,
                 b, tolerance);
}

/// Run all registered tests (or only those whose name contains argv[1]).
inline int run_all(int argc, char** argv) {
    const std::string_view filter = argc > 1 ? std::string_view(argv[1]) : std::string_view();
    int executed = 0;
    for (const TestCase& test : registry()) {
        if (!filter.empty() && std::string_view(test.name).find(filter) == std::string_view::npos) {
            continue;
        }
        const int before = failure_count();
        test.function();
        ++executed;
        std::printf("[%s] %s\n", failure_count() == before ? " OK " : "FAIL", test.name);
    }
    std::printf("%d test(s), %d failure(s)\n", executed, failure_count());
    return failure_count() == 0 && executed > 0 ? 0 : 1;
}

}  // namespace vtt::test

#define VTT_TEST(name)                                                     \
    static void name();                                                    \
    static const ::vtt::test::Registrar name##_registrar{#name, &name};    \
    static void name()

#define CHECK(condition)                                                      \
    do {                                                                      \
        if (!(condition)) {                                                   \
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
