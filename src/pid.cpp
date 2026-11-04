#include "vtt/pid.hpp"

#include <algorithm>

namespace vtt {

Pid::Pid(const PidConfig& config) : config_(config) {}

double Pid::update(double error, double dt) {
    if (dt <= 0.0) {
        return 0.0;
    }
    if (has_previous_) {
        const double raw = (error - previous_error_) / dt;
        const double alpha = dt / (config_.derivative_tau + dt);
        filtered_derivative_ += alpha * (raw - filtered_derivative_);
    }
    previous_error_ = error;
    has_previous_ = true;

    const double proportional = config_.kp * error;
    const double derivative = config_.kd * filtered_derivative_;
    const double candidate = std::clamp(integral_ + error * dt, -config_.integral_limit,
                                        config_.integral_limit);
    const double unsaturated = proportional + config_.ki * candidate + derivative;
    const bool saturated = unsaturated > config_.output_limit || unsaturated < -config_.output_limit;
    const bool pushing_further = (unsaturated > 0.0) == (error > 0.0);
    if (!(saturated && pushing_further)) {
        integral_ = candidate;
    }
    const double output = proportional + config_.ki * integral_ + derivative;
    return std::clamp(output, -config_.output_limit, config_.output_limit);
}

void Pid::reset() noexcept {
    integral_ = 0.0;
    previous_error_ = 0.0;
    filtered_derivative_ = 0.0;
    has_previous_ = false;
}

}  // namespace vtt
