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
- Detect + track: `--detector HOST:PORT` streams frames over UDP to `python/vtt_detector_server.py` (YOLO), whose boxes correct the correlation tracker. `DetectionFusion` auto-locks (`--auto-lock`), re-anchors a drifted target after two disagreeing detections, refines small drift at once with motion compensation for the reply delay, and never overrides a target the detector has not confirmed. Options `--detector-listen`, `--detect-every`, `--det-conf`. Launcher `python/run_detect_track.ps1`.
- Detector round trip: `vtt` keeps the send time of the last 120 frames and shows `rtt NN ms` on the DET status line. The exit summary prints `detector rtt ms` mean/p50/p95 with reply, refine, and re-anchor counts. `capture.read()` is reported as `capture ms` mean/p95.
- Optional inference time on detector replies: `DET <id> <x> <y> <w> <h> <conf> <infer_ms>` and `NONE <id> <infer_ms>`. Negative or non-finite values are rejected. Older 7-field and 2-field lines still parse. The status line shows `yolo NN ms`.
- Live cameras request a one-frame capture buffer (`CAP_PROP_BUFFERSIZE` 1). `--cam-width` and `--cam-height` (`0` = driver default) and `--cam-fps` request a mode; `vtt` prints the size and fps the driver delivered.
- `vtt_detector_server.py`: `--half` / `--no-half` (FP16 on CUDA, default on when CUDA is available), a warm-up inference before `detector ready`, `infer_ms` on each reply, and the same `YOLO()` load for `.pt`, `.onnx`, and `.engine`.

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
