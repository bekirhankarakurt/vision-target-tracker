#include "test_harness.hpp"

#include "vtt/kalman.hpp"

using namespace vtt;

VTT_TEST(kalman_converges_to_constant_velocity) {
    ConstantVelocityKalman kf;
    kf.init({0.0, 0.0});
    cv::RNG rng(4);
    for (int t = 1; t <= 60; ++t) {
        kf.predict(1.0);
        kf.correct({3.0 * t + rng.gaussian(1.0), -2.0 * t + rng.gaussian(1.0)});
    }
    CHECK_NEAR(kf.velocity().x, 3.0, 0.3);
    CHECK_NEAR(kf.velocity().y, -2.0, 0.3);
    CHECK_NEAR(kf.position().x, 180.0, 2.0);
}

VTT_TEST(kalman_predicts_through_gap_and_grows_uncertainty) {
    ConstantVelocityKalman kf;
    kf.init({10.0, 10.0});
    for (int t = 1; t <= 30; ++t) {
        kf.predict(1.0);
        kf.correct({10.0 + 2.0 * t, 10.0});
    }
    const double sigma_before = kf.position_std();
    for (int t = 31; t <= 40; ++t) {
        kf.predict(1.0);
    }
    CHECK_NEAR(kf.position().x, 90.0, 2.0);
    CHECK_NEAR(kf.position().y, 10.0, 1.0);
    CHECK(kf.position_std() > 2.0 * sigma_before);
}

VTT_TEST(kalman_covariance_stays_symmetric) {
    ConstantVelocityKalman kf;
    kf.init({0.0, 0.0});
    for (int t = 1; t <= 100; ++t) {
        kf.predict(0.5);
        kf.correct({t * 1.0, t * 0.5});
    }
    const cv::Matx44d& p = kf.covariance();
    for (int i = 0; i < 4; ++i) {
        CHECK(p(i, i) > 0.0);
        for (int j = 0; j < 4; ++j) {
            CHECK_NEAR(p(i, j), p(j, i), 1e-9);
        }
    }
}

VTT_TEST(kalman_damping_and_set_position) {
    ConstantVelocityKalman kf;
    kf.init({0.0, 0.0});
    for (int t = 1; t <= 20; ++t) {
        kf.predict(1.0);
        kf.correct({4.0 * t, 0.0});
    }
    const double vx = kf.velocity().x;
    kf.damp_velocity(0.5);
    CHECK_NEAR(kf.velocity().x, 0.5 * vx, 1e-12);
    kf.set_position({7.0, 8.0});
    CHECK_NEAR(kf.position().x, 7.0, 1e-12);
    CHECK_NEAR(kf.position().y, 8.0, 1e-12);
}

VTT_TEST_MAIN()
