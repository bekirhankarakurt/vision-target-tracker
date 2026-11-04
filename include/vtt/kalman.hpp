/// @file kalman.hpp
/// @brief Constant-velocity Kalman filter for the target centre.
#pragma once

#include <opencv2/core.hpp>

namespace vtt {

/// Noise parameters of the constant-velocity model (units: pixels, frames).
struct KalmanConfig {
    double process_noise = 0.5;      ///< White-acceleration spectral density q.
    double measurement_noise = 1.5;  ///< Measurement std-dev (px).
    double initial_velocity_std = 10.0;
};

/// State x = [px, py, vx, vy]^T, measurement z = [px, py]^T.
class ConstantVelocityKalman {
public:
    explicit ConstantVelocityKalman(const KalmanConfig& config = {});

    void init(const cv::Point2d& position);
    void predict(double dt);
    void correct(const cv::Point2d& measurement);
    /// Multiply the velocity by @p factor (used while the target is lost).
    void damp_velocity(double factor);
    /// Overwrite the position, keeping the covariance (used for frame clamping).
    void set_position(const cv::Point2d& position);

    [[nodiscard]] cv::Point2d position() const noexcept;
    [[nodiscard]] cv::Point2d velocity() const noexcept;
    [[nodiscard]] const cv::Matx44d& covariance() const noexcept { return covariance_; }
    /// 1-sigma positional uncertainty (sqrt of the mean positional variance).
    [[nodiscard]] double position_std() const noexcept;

private:
    KalmanConfig config_;
    cv::Vec4d state_;
    cv::Matx44d covariance_;
};

}  // namespace vtt
