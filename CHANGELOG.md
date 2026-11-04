# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-11-04

### Added

- MOSSE correlation filter with log/normalise/cosine preprocessing and 8-sample affine initialisation.
- KCF filter with a Gaussian kernel on intensity + Sobel gradient features (no HOG).
- DSST-style 1-D scale filter over a 15-level pyramid.
- Peak-to-sidelobe ratio (PSR) confidence with sub-pixel peak refinement.
- Constant-velocity Kalman filter (Joseph-form update) for prediction through occlusion.
- Tracking / Occluded / Lost state machine with an expanding grid search for reacquisition.
- Simulated 2-axis gimbal: PID loops with derivative filtering, conditional-integration
  anti-windup, rate, acceleration and angle limits; zoomed payload inset view.
- Telemetry overlay (state, PSR, FPS, latency, pan/tilt angles, rates and errors).
- `vtt` application: webcam or video input, mouse or CLI ROI, keyboard controls,
  headless mode, per-frame CSV, overlay video output, accuracy thresholds against ground truth.
- `vtt_synth` deterministic synthetic sequence generator with ground truth
  (textured target, Lissajous motion, scale change, sliding occluder, sensor noise).
- CTest unit and integration tests asserting IoU and centre-error thresholds.
- CI (GCC, Clang, ASan/UBSan, clang-tidy, cppcheck), CodeQL, coverage and Doxygen on GitHub Pages,
  tagged releases with SHA256 checksums.

[Unreleased]: https://github.com/bekirhankarakurt/vision-target-tracker/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/bekirhankarakurt/vision-target-tracker/releases/tag/v1.0.0
