/// @file pid.hpp
/// @brief PID controller with derivative filtering and anti-windup.
#pragma once

namespace vtt {

struct PidConfig {
    double kp = 10.0;
    double ki = 20.0;
    double kd = 0.1;
    double output_limit = 180.0;     ///< |u| saturation.
    double integral_limit = 30.0;    ///< |integral| clamp (error * s).
    double derivative_tau = 0.02;    ///< First-order derivative filter time constant (s).
};

/// Parallel-form PID: u = kp*e + ki*int(e) + kd*de/dt.
///
/// Anti-windup uses conditional integration (the integrator freezes while the
/// output is saturated in the direction of the error) plus a hard clamp.
class Pid {
public:
    explicit Pid(const PidConfig& config = {});

    [[nodiscard]] double update(double error, double dt);
    void reset() noexcept;

    [[nodiscard]] double integral() const noexcept { return integral_; }
    [[nodiscard]] const PidConfig& config() const noexcept { return config_; }
    void set_config(const PidConfig& config) noexcept { config_ = config; }

private:
    PidConfig config_;
    double integral_ = 0.0;
    double previous_error_ = 0.0;
    double filtered_derivative_ = 0.0;
    bool has_previous_ = false;
};

}  // namespace vtt
