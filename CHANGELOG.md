# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Python companion `python/vtt_detect.py` for YOLO detection and optional Orbbec depth.
  The on-screen distance is the median of valid depth pixels in the central 40% of each box.
  `pyorbbecsdk2` is required only for Orbbec cameras. See `python/README.md`.
- `python/vtt_custom.py`: captures sessions, labels them with SAM 2, builds leak-free splits and trains a YOLO segmentation model for a custom object.
  Detections also draw segmentation masks.

### Changed

- GUI target selection runs on the live video. The window opens immediately with the tracker `IDLE` and no gimbal aim until a drag (at least 8x8 px) or a click locks a target; right-click or `x` releases it, and a new gesture switches targets. `--click-box` sets the click square (default 80, range 16..400). `--roi` / `--gt` still initialise immediately, including in `--headless`.

### Fixed

- `unit.utils` crashed on Windows because it deleted a CSV file that was still open.

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
