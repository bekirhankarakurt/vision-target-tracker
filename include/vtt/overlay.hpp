/// @file overlay.hpp
/// @brief Telemetry overlay rendering (boxes, reticle, gimbal inset, HUD).
#pragma once

#include "vtt/gimbal.hpp"
#include "vtt/tracker.hpp"
#include "vtt/types.hpp"

#include <opencv2/core.hpp>

namespace vtt {

struct Telemetry {
    Algorithm algorithm = Algorithm::Kcf;
    long frame_index = 0;
    double fps = 0.0;
    double latency_ms = 0.0;
    bool paused = false;
};

struct OverlayOptions {
    bool show_gimbal_view = true;
    bool show_help = false;
};

/// Draw the full overlay onto @p canvas (8-bit BGR) in place.
void draw_overlay(cv::Mat& canvas, const TrackOutput& track, const GimbalSim& gimbal,
                  const Telemetry& telemetry, const OverlayOptions& options);

/// BGR colour associated with a tracker state.
[[nodiscard]] cv::Scalar state_color(TrackState state) noexcept;

}  // namespace vtt
