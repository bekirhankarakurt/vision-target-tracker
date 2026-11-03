#include "test_harness.hpp"

#include "vtt/scale_filter.hpp"

#include <opencv2/imgproc.hpp>

using namespace vtt;

namespace {

/// Gray frame with a textured square of side @p side centred at (160, 120).
cv::Mat frame_with_target(double side) {
    cv::Mat texture(64, 64, CV_8U);
    cv::RNG rng(21);
    rng.fill(texture, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(texture, texture, cv::Size(), 1.5);
    cv::Mat frame(240, 320, CV_8U, cv::Scalar(90));
    const int s = cvRound(side);
    cv::Mat resized;
    cv::resize(texture, resized, cv::Size(s, s), 0.0, 0.0, cv::INTER_AREA);
    resized.copyTo(frame(cv::Rect(160 - s / 2, 120 - s / 2, s, s)));
    return frame;
}

}  // namespace

VTT_TEST(scale_filter_detects_growth_and_shrink) {
    ScaleFilter filter;
    const cv::Point2d center(160.0, 120.0);
    filter.init(frame_with_target(60.0), center, cv::Size2d(60.0, 60.0));
    CHECK(filter.initialized());
    CHECK_NEAR(filter.estimate(frame_with_target(60.0), center, cv::Size2d(60.0, 60.0)), 1.0, 0.021);
    // As in the tracker, the estimate is applied frame after frame and converges.
    for (const double true_side : {66.0, 54.0}) {
        const cv::Mat frame = frame_with_target(true_side);
        double side = 60.0;
        for (int i = 0; i < 15; ++i) {
            side *= filter.estimate(frame, center, cv::Size2d(side, side));
        }
        CHECK_NEAR(side, true_side, 0.03 * true_side);
    }
}

VTT_TEST(scale_filter_factors_are_symmetric) {
    const ScaleFilter filter(ScaleFilterParams{});
    const auto& f = filter.factors();
    CHECK(f.size() == 17U);
    CHECK_NEAR(f[8], 1.0, 1e-12);
    CHECK_NEAR(f.front() * f.back(), 1.0, 1e-12);
}

VTT_TEST(scale_filter_rejects_even_levels) {
    ScaleFilterParams params;
    params.num_scales = 4;
    bool threw = false;
    try {
        const ScaleFilter filter(params);
        static_cast<void>(filter);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

VTT_TEST_MAIN()
