/// @file cli.hpp
/// @brief Command-line parsing for the vtt application.
#pragma once

#include "vtt/gimbal.hpp"
#include "vtt/types.hpp"

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

namespace vtt {

struct AppOptions {
    std::optional<std::string> input;   ///< Video file; camera when empty.
    int camera_index = 0;
    Algorithm algorithm = Algorithm::Kcf;
    std::optional<cv::Rect2d> roi;
    std::optional<std::string> ground_truth;
    std::optional<std::string> csv_path;
    std::optional<std::string> output_video;
    bool headless = false;
    bool disable_scale = false;
    long max_frames = 0;                ///< 0 = unlimited.
    std::optional<double> psr_threshold;
    GimbalConfig gimbal{};
    std::optional<double> min_iou;      ///< Exit non-zero if mean IoU is lower.
    std::optional<double> max_center_error;
    bool show_help = false;
    bool show_version = false;
};

struct ParseResult {
    AppOptions options;
    std::string error;  ///< Non-empty on failure.
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

[[nodiscard]] ParseResult parse_arguments(const std::vector<std::string>& args);
[[nodiscard]] std::string usage();

/// Parse "x,y,w,h" into a rectangle (w, h > 0).
[[nodiscard]] std::optional<cv::Rect2d> parse_rect(const std::string& text);

}  // namespace vtt
