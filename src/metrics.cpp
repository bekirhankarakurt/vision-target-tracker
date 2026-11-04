#include "vtt/metrics.hpp"

#include <algorithm>
#include <cmath>

namespace vtt {

namespace {
constexpr double kSuccessIou = 0.5;
constexpr double kPrecisionPx = 20.0;
}  // namespace

double iou(const cv::Rect2d& a, const cv::Rect2d& b) noexcept {
    const double inter = (a & b).area();
    const double uni = a.area() + b.area() - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

double center_error(const cv::Rect2d& a, const cv::Rect2d& b) noexcept {
    const double dx = (a.x + 0.5 * a.width) - (b.x + 0.5 * b.width);
    const double dy = (a.y + 0.5 * a.height) - (b.y + 0.5 * b.height);
    return std::hypot(dx, dy);
}

void AccuracyStats::add(const cv::Rect2d& estimate, const cv::Rect2d& truth) {
    const double overlap = iou(estimate, truth);
    const double error = center_error(estimate, truth);
    ++count_;
    iou_sum_ += overlap;
    error_sum_ += error;
    successes_ += overlap >= kSuccessIou ? 1U : 0U;
    precise_ += error <= kPrecisionPx ? 1U : 0U;
}

double AccuracyStats::mean_iou() const noexcept {
    return count_ > 0 ? iou_sum_ / static_cast<double>(count_) : 0.0;
}

double AccuracyStats::mean_center_error() const noexcept {
    return count_ > 0 ? error_sum_ / static_cast<double>(count_) : 0.0;
}

double AccuracyStats::success_rate() const noexcept {
    return count_ > 0 ? static_cast<double>(successes_) / static_cast<double>(count_) : 0.0;
}

double AccuracyStats::precision_20px() const noexcept {
    return count_ > 0 ? static_cast<double>(precise_) / static_cast<double>(count_) : 0.0;
}

void LatencyStats::add(double milliseconds) {
    samples_.push_back(milliseconds);
    sum_ += milliseconds;
    max_ = std::max(max_, milliseconds);
}

double LatencyStats::mean() const noexcept {
    return samples_.empty() ? 0.0 : sum_ / static_cast<double>(samples_.size());
}

double LatencyStats::max() const noexcept { return max_; }

double LatencyStats::percentile(double p) const {
    if (samples_.empty()) {
        return 0.0;
    }
    std::vector<double> sorted = samples_;
    std::sort(sorted.begin(), sorted.end());
    const double rank = std::ceil(std::clamp(p, 0.0, 100.0) / 100.0 *
                                  static_cast<double>(sorted.size()));
    const auto index = static_cast<std::size_t>(std::max(rank, 1.0)) - 1U;
    return sorted[std::min(index, sorted.size() - 1U)];
}

void RateMeter::tick(double seconds) noexcept {
    if (seconds <= 0.0) {
        return;
    }
    const double instant = 1.0 / seconds;
    rate_ = rate_ <= 0.0 ? instant : rate_ + smoothing_ * (instant - rate_);
}

}  // namespace vtt
