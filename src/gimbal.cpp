#include "vtt/gimbal.hpp"

#include <algorithm>
#include <cmath>

namespace vtt {

namespace {

constexpr double kDegToRad = CV_PI / 180.0;
constexpr double kRadToDeg = 180.0 / CV_PI;

}  // namespace

GimbalSim::GimbalSim(const GimbalConfig& config, cv::Size frame_size)
    : config_(config),
      frame_size_(frame_size),
      focal_px_(0.5 * static_cast<double>(frame_size.width) /
                std::tan(0.5 * config.horizontal_fov_deg * kDegToRad)),
      pan_pid_(config.pan_pid),
      tilt_pid_(config.tilt_pid) {
    // The controller output is a rate command: saturating it at the axis rate
    // limit keeps the conditional-integration anti-windup aware of the actuator.
    PidConfig pan = config.pan_pid;
    PidConfig tilt = config.tilt_pid;
    pan.output_limit = std::min(pan.output_limit, config.pan_limits.max_rate_dps);
    tilt.output_limit = std::min(tilt.output_limit, config.tilt_limits.max_rate_dps);
    pan_pid_.set_config(pan);
    tilt_pid_.set_config(tilt);
    CV_Assert(frame_size.width > 0 && frame_size.height > 0);
    CV_Assert(config.horizontal_fov_deg > 0.0 && config.horizontal_fov_deg < 180.0);
}

Angles GimbalSim::pixel_to_angles(const cv::Point2d& pixel) const {
    const double dx = pixel.x - 0.5 * static_cast<double>(frame_size_.width);
    const double dy = 0.5 * static_cast<double>(frame_size_.height) - pixel.y;
    return {std::atan2(dx, focal_px_) * kRadToDeg, std::atan2(dy, focal_px_) * kRadToDeg};
}

cv::Point2d GimbalSim::angles_to_pixel(const Angles& angles) const {
    return {0.5 * static_cast<double>(frame_size_.width) +
                focal_px_ * std::tan(angles.pan_deg * kDegToRad),
            0.5 * static_cast<double>(frame_size_.height) -
                focal_px_ * std::tan(angles.tilt_deg * kDegToRad)};
}

cv::Point2d GimbalSim::boresight_pixel() const {
    return angles_to_pixel({pan_.angle, tilt_.angle});
}

void GimbalSim::advance(Axis& axis, double rate_command, const AxisLimits& limits, double dt) {
    const double command = std::clamp(rate_command, -limits.max_rate_dps, limits.max_rate_dps);
    const double max_delta = limits.max_accel_dps2 * dt;
    axis.rate += std::clamp(command - axis.rate, -max_delta, max_delta);
    axis.angle += axis.rate * dt;
    if (axis.angle > limits.max_deg) {
        axis.angle = limits.max_deg;
        axis.rate = std::min(axis.rate, 0.0);
    } else if (axis.angle < limits.min_deg) {
        axis.angle = limits.min_deg;
        axis.rate = std::max(axis.rate, 0.0);
    }
}

GimbalState GimbalSim::step(const std::optional<cv::Point2d>& target_px, double dt) {
    if (dt <= 0.0) {
        return state_;
    }
    double pan_command = 0.0;
    double tilt_command = 0.0;
    if (target_px) {
        const Angles bearing = pixel_to_angles(*target_px);
        state_.pan_error_deg = bearing.pan_deg - pan_.angle;
        state_.tilt_error_deg = bearing.tilt_deg - tilt_.angle;
        pan_command = pan_pid_.update(state_.pan_error_deg, dt);
        tilt_command = tilt_pid_.update(state_.tilt_error_deg, dt);
    } else {
        pan_pid_.reset();
        tilt_pid_.reset();
        state_.pan_error_deg = 0.0;
        state_.tilt_error_deg = 0.0;
    }
    advance(pan_, pan_command, config_.pan_limits, dt);
    advance(tilt_, tilt_command, config_.tilt_limits, dt);
    state_.pan_deg = pan_.angle;
    state_.tilt_deg = tilt_.angle;
    state_.pan_rate_dps = pan_.rate;
    state_.tilt_rate_dps = tilt_.rate;
    state_.has_target = target_px.has_value();
    return state_;
}

void GimbalSim::reset() {
    pan_ = {};
    tilt_ = {};
    pan_pid_.reset();
    tilt_pid_.reset();
    state_ = {};
}

}  // namespace vtt
