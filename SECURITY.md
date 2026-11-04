# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | Yes       |
| < 1.0   | No        |

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/bekirhankarakurt/vision-target-tracker/security/advisories/new)
instead and include:

- affected version (`vtt --version`) and platform,
- a minimal input (video file, ground-truth CSV or command line) that triggers the issue,
- the observed impact (crash, memory corruption, out-of-bounds read, ...).

You can expect an acknowledgement within 7 days and a fix or mitigation plan within 30 days.

## Scope

The project parses untrusted inputs in two places: video files (decoded by OpenCV/FFmpeg)
and CSV ground-truth files (`vtt/csv.hpp`). Issues in OpenCV or FFmpeg themselves should be
reported upstream. All code is continuously checked with AddressSanitizer, UBSan,
clang-tidy, cppcheck and CodeQL.
