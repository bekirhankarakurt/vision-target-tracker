# Contributing

Thanks for your interest in improving vision-target-tracker.

## Development setup

```bash
sudo apt-get install -y build-essential cmake libopencv-dev
cmake -S . -B build -DVTT_WARNINGS_AS_ERRORS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Before opening a pull request, also run the sanitizer build and the static analysers used in CI:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DVTT_ENABLE_SANITIZERS=ON
cmake --build build-asan -j && ctest --test-dir build-asan --output-on-failure
cmake -S . -B build-tidy -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
run-clang-tidy -p build-tidy -quiet "$PWD/(src|apps|tests)/"
cppcheck --enable=warning,style,performance,portability --std=c++17 --error-exitcode=1 \
  --suppress=missingIncludeSystem --quiet -I include src apps tests
```

## Guidelines

- C++17, RAII, no raw `new`/`delete`, `const` wherever possible, `[[nodiscard]]` on queries.
- Zero warnings with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`.
- OpenCV (`core`, `imgproc`, `videoio`, `highgui`) is the only dependency; tracking maths stays in this repository.
- Every behavioural change comes with a test in `tests/`. Tracker changes must keep the
  synthetic-sequence thresholds (`integration.headless_*`) green and should report the
  latency impact from `vtt --headless`.
- Small, focused commits with imperative subject lines ("Add ...", "Fix ...").
- Update `CHANGELOG.md` under *Unreleased*.
