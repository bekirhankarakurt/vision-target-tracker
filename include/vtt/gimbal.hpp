/// @file gimbal.hpp
/// @brief Simulated 2-axis pan/tilt gimbal driven by PID loops.
#pragma once

#include "vtt/pid.hpp"

#include <opencv2/core.hpp>

#include <optional>

namespace vtt {

/// Mechanical limits of one gimbal axis.
struct AxisLimits {
    double min_deg = -170.0;
    double max_deg = 170.0;
    double max_rate_dps = 120.0;    ///< Rate limit (deg/s).
    double max_accel_dps2 = 600.0;  ///< Acceleration limit (deg/s^2).
};

struct GimbalConfig {
    double horizontal_fov_deg = 60.0;  ///< FOV of the wide (scene) camera.
    double zoom = 4.0;                 ///< Narrow-FOV payload magnification for the inset view.
    PidConfig pan_pid{};
    PidConfig tilt_pid{};
    AxisLimits pan_limits{};
    AxisLimits tilt_limits{-45.0, 45.0, 120.0, 600.0};
};

/// Instantaneous gimbal telemetry.
struct GimbalState {
    double pan_deg = 0.0;
    double tilt_deg = 0.0;
    double pan_rate_dps = 0.0;
    double tilt_rate_dps = 0.0;
    double pan_error_deg = 0.0;
    double tilt_error_deg = 0.0;
    bool has_target = false;
};

/// Pan/tilt angles of a line of sight.
struct Angles {
    double pan_deg = 0.0;
    double tilt_deg = 0.0;
};

/// Gimbal carrying a narrow-FOV payload, mounted at the scene camera.
/// The loops drive the boresight onto the target bearing measured in the
/// wide camera image (pinhole model).
class GimbalSim {
public:
    GimbalSim(const GimbalConfig& config, cv::Size frame_size);

    /// Advance the simulation by @p dt seconds. With no target the axes decelerate to rest.
    GimbalState step(const std::optional<cv::Point2d>& target_px, double dt);
    void reset();

    [[nodiscard]] Angles pixel_to_angles(const cv::Point2d& pixel) const;
    [[nodiscard]] cv::Point2d angles_to_pixel(const Angles& angles) const;
    [[nodiscard]] cv::Point2d boresight_pixel() const;
    [[nodiscard]] const GimbalState& state() const noexcept { return state_; }
    [[nodiscard]] const GimbalConfig& config() const noexcept { return config_; }
    [[nodiscard]] double focal_length_px() const noexcept { return focal_px_; }

private:
    struct Axis {
        double angle = 0.0;
        double rate = 0.0;
    };
    static void advance(Axis& axis, double rate_command, const AxisLimits& limits, double dt);

    GimbalConfig config_;
    cv::Size frame_size_;
    double focal_px_;
    Pid pan_pid_;
    Pid tilt_pid_;
    Axis pan_;
    Axis tilt_;
    GimbalState state_;
};

}  // namespace vtt
