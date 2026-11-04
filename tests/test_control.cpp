#include "test_harness.hpp"

#include "vtt/gimbal.hpp"
#include "vtt/pid.hpp"

#include <cmath>

using namespace vtt;

VTT_TEST(pid_proportional_only) {
    Pid pid({2.0, 0.0, 0.0, 100.0, 10.0, 0.01});
    CHECK_NEAR(pid.update(3.0, 0.01), 6.0, 1e-12);
    CHECK_NEAR(pid.update(-1.0, 0.01), -2.0, 1e-12);
}

VTT_TEST(pid_output_is_saturated) {
    Pid pid({100.0, 0.0, 0.0, 5.0, 10.0, 0.01});
    CHECK_NEAR(pid.update(1.0, 0.01), 5.0, 1e-12);
    CHECK_NEAR(pid.update(-1.0, 0.01), -5.0, 1e-12);
}

VTT_TEST(pid_anti_windup_freezes_integral_when_saturated) {
    Pid pid({10.0, 5.0, 0.0, 20.0, 100.0, 0.01});
    for (int i = 0; i < 500; ++i) {
        CHECK_NEAR(pid.update(10.0, 0.01), 20.0, 1e-9);
    }
    // P alone saturates, so the integrator must not have accumulated.
    CHECK_NEAR(pid.integral(), 0.0, 1e-9);
    // Recovery is immediate once the error reverses.
    CHECK(pid.update(-1.0, 0.01) < 0.0);
}

VTT_TEST(pid_integral_is_clamped) {
    Pid pid({0.0, 1.0, 0.0, 1000.0, 2.0, 0.01});
    for (int i = 0; i < 1000; ++i) {
        static_cast<void>(pid.update(1.0, 0.01));
    }
    CHECK_NEAR(pid.integral(), 2.0, 1e-9);
    pid.reset();
    CHECK_NEAR(pid.integral(), 0.0, 1e-12);
}

VTT_TEST(pid_rejects_non_positive_dt) {
    Pid pid;
    CHECK_NEAR(pid.update(1.0, 0.0), 0.0, 1e-12);
}

VTT_TEST(gimbal_pixel_angle_roundtrip) {
    const GimbalSim gimbal(GimbalConfig{}, cv::Size(640, 480));
    const Angles center = gimbal.pixel_to_angles({320.0, 240.0});
    CHECK_NEAR(center.pan_deg, 0.0, 1e-12);
    CHECK_NEAR(center.tilt_deg, 0.0, 1e-12);
    const Angles edge = gimbal.pixel_to_angles({640.0, 240.0});
    CHECK_NEAR(edge.pan_deg, 30.0, 1e-9);
    const cv::Point2d p = gimbal.angles_to_pixel(gimbal.pixel_to_angles({100.0, 400.0}));
    CHECK_NEAR(p.x, 100.0, 1e-9);
    CHECK_NEAR(p.y, 400.0, 1e-9);
}

VTT_TEST(gimbal_converges_on_static_target_without_overshoot_beyond_limit) {
    GimbalSim gimbal(GimbalConfig{}, cv::Size(640, 480));
    const cv::Point2d target(560.0, 120.0);
    const Angles bearing = gimbal.pixel_to_angles(target);
    double max_pan = 0.0;
    for (int i = 0; i < 300; ++i) {
        const GimbalState s = gimbal.step(target, 1.0 / 100.0);
        max_pan = std::max(max_pan, s.pan_deg);
    }
    CHECK_NEAR(gimbal.state().pan_deg, bearing.pan_deg, 0.05);
    CHECK_NEAR(gimbal.state().tilt_deg, bearing.tilt_deg, 0.05);
    CHECK(max_pan < bearing.pan_deg * 1.15);
    const cv::Point2d bore = gimbal.boresight_pixel();
    CHECK_NEAR(bore.x, target.x, 0.5);
    CHECK_NEAR(bore.y, target.y, 0.5);
}

VTT_TEST(gimbal_respects_rate_and_acceleration_limits) {
    GimbalConfig config;
    config.pan_limits.max_rate_dps = 20.0;
    config.pan_limits.max_accel_dps2 = 100.0;
    GimbalSim gimbal(config, cv::Size(640, 480));
    double previous_rate = 0.0;
    const double dt = 0.01;
    for (int i = 0; i < 200; ++i) {
        const GimbalState s = gimbal.step(cv::Point2d(639.0, 240.0), dt);
        CHECK(std::abs(s.pan_rate_dps) <= 20.0 + 1e-9);
        CHECK(std::abs(s.pan_rate_dps - previous_rate) <= 100.0 * dt + 1e-9);
        previous_rate = s.pan_rate_dps;
    }
}

VTT_TEST(gimbal_respects_angle_limits_and_stops_without_target) {
    GimbalConfig config;
    config.horizontal_fov_deg = 170.0;
    config.tilt_limits.max_deg = 5.0;
    GimbalSim gimbal(config, cv::Size(640, 480));
    for (int i = 0; i < 500; ++i) {
        static_cast<void>(gimbal.step(cv::Point2d(320.0, 0.0), 0.01));
    }
    CHECK_NEAR(gimbal.state().tilt_deg, 5.0, 1e-9);
    for (int i = 0; i < 500; ++i) {
        static_cast<void>(gimbal.step(std::nullopt, 0.01));
    }
    CHECK_NEAR(gimbal.state().pan_rate_dps, 0.0, 1e-9);
    CHECK_NEAR(gimbal.state().tilt_rate_dps, 0.0, 1e-9);
    CHECK(!gimbal.state().has_target);
    gimbal.reset();
    CHECK_NEAR(gimbal.state().tilt_deg, 0.0, 1e-12);
}

VTT_TEST_MAIN()
