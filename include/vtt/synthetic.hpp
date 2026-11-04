/// @file synthetic.hpp
/// @brief Deterministic synthetic test sequence with ground truth: a textured
///        target on a textured background, moving along a Lissajous path with
///        scale change and an occluder that slides across it.
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>

namespace vtt {

struct SyntheticConfig {
    cv::Size frame_size{640, 480};
    int num_frames = 300;
    double fps = 30.0;
    cv::Size target_size{64, 48};   ///< Size at scale 1.
    double scale_amplitude = 0.3;   ///< Scale varies in [1 - a, 1 + a].
    cv::Point2d amplitude{200.0, 120.0};
    double period_x = 240.0;        ///< Frames per horizontal oscillation.
    double period_y = 170.0;        ///< Frames per vertical oscillation.
    double scale_period = 300.0;    ///< Frames per scale oscillation.
    int occlusion_start = 120;      ///< First frame the occluder overlaps the target.
    int occlusion_ramp = 8;         ///< Frames to slide in (and out).
    int occlusion_hold = 10;        ///< Frames of full cover.
    double noise_std = 4.0;         ///< Additive sensor noise (gray levels).
    std::uint64_t seed = 7;
};

struct GroundTruth {
    cv::Rect2d box;
    double visibility = 1.0;  ///< Visible fraction of the target area in [0, 1].
};

class SyntheticScene {
public:
    explicit SyntheticScene(const SyntheticConfig& config = {});

    [[nodiscard]] cv::Mat render(int frame) const;  ///< 8-bit BGR frame.
    [[nodiscard]] GroundTruth ground_truth(int frame) const;
    [[nodiscard]] const SyntheticConfig& config() const noexcept { return config_; }

private:
    [[nodiscard]] cv::Rect2d target_box(int frame) const;
    [[nodiscard]] cv::Rect2d occluder_box(int frame) const;  ///< Empty when inactive.

    SyntheticConfig config_;
    cv::Mat background_;
    cv::Mat target_texture_;
    cv::Mat occluder_texture_;
};

}  // namespace vtt
