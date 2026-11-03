#include "vtt/kalman.hpp"

#include <cmath>

namespace vtt {

ConstantVelocityKalman::ConstantVelocityKalman(const KalmanConfig& config)
    : config_(config), state_(0.0, 0.0, 0.0, 0.0), covariance_(cv::Matx44d::eye()) {}

void ConstantVelocityKalman::init(const cv::Point2d& position) {
    state_ = cv::Vec4d(position.x, position.y, 0.0, 0.0);
    const double r2 = config_.measurement_noise * config_.measurement_noise;
    const double v2 = config_.initial_velocity_std * config_.initial_velocity_std;
    covariance_ = cv::Matx44d::diag(cv::Vec4d(r2, r2, v2, v2));
}

void ConstantVelocityKalman::predict(double dt) {
    const cv::Matx44d f(1.0, 0.0, dt, 0.0,
                        0.0, 1.0, 0.0, dt,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0);
    // Discrete white-noise acceleration model.
    const double q = config_.process_noise;
    const double dt2 = dt * dt;
    const double a = q * dt2 * dt2 / 4.0;
    const double b = q * dt2 * dt / 2.0;
    const double c = q * dt2;
    const cv::Matx44d process(a, 0.0, b, 0.0,
                              0.0, a, 0.0, b,
                              b, 0.0, c, 0.0,
                              0.0, b, 0.0, c);
    state_ = f * state_;
    covariance_ = f * covariance_ * f.t() + process;
}

void ConstantVelocityKalman::correct(const cv::Point2d& measurement) {
    const cv::Matx<double, 2, 4> h(1.0, 0.0, 0.0, 0.0,
                                   0.0, 1.0, 0.0, 0.0);
    const double r2 = config_.measurement_noise * config_.measurement_noise;
    const cv::Matx22d r(r2, 0.0, 0.0, r2);
    const cv::Vec2d innovation(measurement.x - state_[0], measurement.y - state_[1]);
    const cv::Matx22d s = h * covariance_ * h.t() + r;
    const cv::Matx<double, 4, 2> gain = covariance_ * h.t() * s.inv();
    state_ += gain * innovation;
    // Joseph form keeps the covariance symmetric positive definite.
    const cv::Matx44d i_kh = cv::Matx44d::eye() - gain * h;
    covariance_ = i_kh * covariance_ * i_kh.t() + gain * r * gain.t();
}

void ConstantVelocityKalman::damp_velocity(double factor) {
    state_[2] *= factor;
    state_[3] *= factor;
}

void ConstantVelocityKalman::set_position(const cv::Point2d& position) {
    state_[0] = position.x;
    state_[1] = position.y;
}

cv::Point2d ConstantVelocityKalman::position() const noexcept { return {state_[0], state_[1]}; }

cv::Point2d ConstantVelocityKalman::velocity() const noexcept { return {state_[2], state_[3]}; }

double ConstantVelocityKalman::position_std() const noexcept {
    return std::sqrt(0.5 * (covariance_(0, 0) + covariance_(1, 1)));
}

}  // namespace vtt
