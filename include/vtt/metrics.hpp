/// @file metrics.hpp
/// @brief Tracking accuracy and latency statistics.
#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace vtt {

/// Intersection over union of two boxes (0 when either is empty).
[[nodiscard]] double iou(const cv::Rect2d& a, const cv::Rect2d& b) noexcept;

/// Euclidean distance between box centres (px).
[[nodiscard]] double center_error(const cv::Rect2d& a, const cv::Rect2d& b) noexcept;

/// Accumulates per-frame accuracy against ground truth.
class AccuracyStats {
public:
    void add(const cv::Rect2d& estimate, const cv::Rect2d& truth);

    [[nodiscard]] std::size_t count() const noexcept { return count_; }
    [[nodiscard]] double mean_iou() const noexcept;
    [[nodiscard]] double mean_center_error() const noexcept;
    /// Fraction of frames with IoU >= 0.5.
    [[nodiscard]] double success_rate() const noexcept;
    /// Fraction of frames with centre error <= 20 px.
    [[nodiscard]] double precision_20px() const noexcept;

private:
    std::size_t count_ = 0;
    std::size_t successes_ = 0;
    std::size_t precise_ = 0;
    double iou_sum_ = 0.0;
    double error_sum_ = 0.0;
};

/// Collects per-frame latencies (ms) and reports summary percentiles.
class LatencyStats {
public:
    void add(double milliseconds);

    [[nodiscard]] std::size_t count() const noexcept { return samples_.size(); }
    [[nodiscard]] double mean() const noexcept;
    [[nodiscard]] double max() const noexcept;
    /// Nearest-rank percentile, @p p in [0, 100].
    [[nodiscard]] double percentile(double p) const;

private:
    std::vector<double> samples_;
    double sum_ = 0.0;
    double max_ = 0.0;
};

/// Exponential moving average of a frame rate.
class RateMeter {
public:
    explicit RateMeter(double smoothing = 0.1) noexcept : smoothing_(smoothing) {}
    void tick(double seconds) noexcept;
    [[nodiscard]] double rate() const noexcept { return rate_; }

private:
    double smoothing_;
    double rate_ = 0.0;
};

}  // namespace vtt
