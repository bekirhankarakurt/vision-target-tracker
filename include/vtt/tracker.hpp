/// @file tracker.hpp
/// @brief Full tracking pipeline: correlation filter, scale pyramid, PSR
///        confidence, Kalman prediction and lost/reacquire logic.
#pragma once

#include "vtt/correlation_filter.hpp"
#include "vtt/kalman.hpp"
#include "vtt/scale_filter.hpp"
#include "vtt/types.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace vtt {

struct TrackerConfig {
    Algorithm algorithm = Algorithm::Kcf;
    FilterParams filter = default_filter_params(Algorithm::Kcf);
    KalmanConfig kalman{};
    double padding = 2.5;            ///< Search window = padding * target size.
    int template_size = 64;          ///< sqrt(area) of the filter patch in pixels.
    bool estimate_scale = true;      ///< Run the 1-D scale filter after each confident detection.
    ScaleFilterParams scale{};
    double min_target_px = 8.0;      ///< Smallest allowed target side.
    double psr_track = 5.5;          ///< Below this a tracked target becomes occluded.
    double psr_update = 7.0;         ///< Minimum PSR to update the appearance model.
    double psr_reacquire = 7.5;      ///< Required PSR to leave Occluded/Lost.
    int max_occluded_frames = 10;    ///< Occluded longer than this -> Lost.
    double search_growth = 0.15;     ///< Lost search radius growth per frame (window units).
    double max_search_radius = 1.5;  ///< Lost search radius cap (window units).
    double search_spacing = 0.5;     ///< Lost search grid spacing (window units).
    double lost_velocity_damping = 0.9;
};

/// Defaults tuned per algorithm.
[[nodiscard]] TrackerConfig default_tracker_config(Algorithm algorithm);

/// Result of one tracker update.
struct TrackOutput {
    TrackState state = TrackState::Idle;
    cv::Rect2d box;             ///< Measured box (Tracking) or predicted box otherwise.
    cv::Point2d predicted;      ///< Kalman prior for this frame.
    cv::Point2d velocity;       ///< Kalman velocity estimate (px/frame).
    cv::Rect2d search_region;   ///< Union of all evaluated search windows.
    double psr = 0.0;
    double scale = 1.0;         ///< Current size relative to the initial box.
    int frames_without_measurement = 0;
};

class Tracker {
public:
    explicit Tracker(const TrackerConfig& config);

    /// Initialise on @p frame (BGR or grayscale, 8-bit) with target @p box.
    void init(const cv::Mat& frame, const cv::Rect2d& box);
    /// Process the next frame.
    [[nodiscard]] TrackOutput update(const cv::Mat& frame);
    void reset();

    [[nodiscard]] TrackState state() const noexcept { return state_; }
    [[nodiscard]] const TrackerConfig& config() const noexcept { return config_; }

private:
    struct Candidate {
        cv::Point2d center;
        double psr = 0.0;
        cv::Rect2d window;
    };

    [[nodiscard]] cv::Size2d window_size(double scale) const;
    [[nodiscard]] cv::Mat extract(const cv::Mat& gray, const cv::Point2d& center, double scale,
                                  cv::Rect& used_window) const;
    [[nodiscard]] Candidate evaluate(const cv::Mat& gray, const cv::Point2d& center) const;
    [[nodiscard]] cv::Size2d target_size() const;
    [[nodiscard]] std::vector<cv::Point2d> search_centers(const cv::Point2d& predicted) const;
    [[nodiscard]] cv::Point2d clamp_to_frame(const cv::Point2d& p) const;

    TrackerConfig config_;
    std::unique_ptr<CorrelationFilter> filter_;
    ScaleFilter scale_filter_;
    ConstantVelocityKalman kalman_;
    TrackState state_ = TrackState::Idle;
    cv::Size frame_size_;
    cv::Size2d base_target_size_;
    cv::Point2d center_;
    double scale_ = 1.0;
    int frames_without_measurement_ = 0;
    int frames_lost_ = 0;
};

/// Convert an 8-bit BGR/BGRA/gray frame to 8-bit grayscale (no copy for gray input).
[[nodiscard]] cv::Mat to_gray(const cv::Mat& frame);

}  // namespace vtt
