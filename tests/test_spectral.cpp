#include "test_harness.hpp"

#include "vtt/spectral.hpp"

#include <opencv2/imgproc.hpp>

using namespace vtt;

VTT_TEST(gaussian_peak_is_centered_and_normalized) {
    const cv::Mat g = spectral::gaussian_peak(cv::Size(32, 24), 2.0);
    double max_value = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(g, nullptr, &max_value, nullptr, &max_loc);
    CHECK_NEAR(max_value, 1.0, 1e-6);
    CHECK(max_loc == cv::Point(16, 12));
}

VTT_TEST(fft_roundtrip_recovers_signal) {
    cv::Mat signal(20, 30, CV_32F);
    cv::RNG rng(3);
    rng.fill(signal, cv::RNG::UNIFORM, -1.0, 1.0);
    const cv::Mat back = spectral::ifft_real(spectral::fft(signal));
    CHECK(back.size() == signal.size());
    CHECK_NEAR(cv::norm(back, signal, cv::NORM_INF), 0.0, 1e-5);
}

VTT_TEST(complex_divide_inverts_multiply) {
    cv::Mat a(8, 8, CV_32FC2);
    cv::Mat b(8, 8, CV_32FC2);
    cv::RNG rng(5);
    rng.fill(a, cv::RNG::UNIFORM, -2.0, 2.0);
    rng.fill(b, cv::RNG::UNIFORM, 0.5, 2.0);
    const cv::Mat product = spectral::multiply(a, b);
    const cv::Mat quotient = spectral::divide(product, b);
    CHECK_NEAR(cv::norm(quotient, a, cv::NORM_INF), 0.0, 1e-4);
}

VTT_TEST(energy_matches_parseval) {
    cv::Mat signal(16, 16, CV_32F);
    cv::RNG rng(9);
    rng.fill(signal, cv::RNG::NORMAL, 0.0, 1.0);
    const double spatial = cv::norm(signal, cv::NORM_L2SQR);
    CHECK_NEAR(spectral::energy(spectral::fft(signal)) / 256.0, spatial, 1e-3 * spatial);
}

VTT_TEST(find_peak_refines_subpixel_location) {
    cv::Mat response(32, 32, CV_32F);
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            const double dx = x - 10.3;
            const double dy = y - 20.6;
            response.at<float>(y, x) = static_cast<float>(5.0 - 0.5 * (dx * dx + dy * dy));
        }
    }
    const spectral::Peak peak = spectral::find_peak(response);
    CHECK(peak.integer_location == cv::Point(10, 21));
    CHECK_NEAR(peak.location.x, 10.3, 1e-3);
    CHECK_NEAR(peak.location.y, 20.6, 1e-3);
}

VTT_TEST(psr_separates_sharp_peak_from_noise) {
    cv::Mat noise(64, 64, CV_32F);
    cv::RNG rng(11);
    rng.fill(noise, cv::RNG::NORMAL, 0.0, 0.05);
    cv::Mat sharp = noise + spectral::gaussian_peak(noise.size(), 2.0);
    const double psr_sharp = spectral::peak_to_sidelobe_ratio(sharp, spectral::find_peak(sharp));
    const double psr_noise = spectral::peak_to_sidelobe_ratio(noise, spectral::find_peak(noise));
    CHECK(psr_sharp > 15.0);
    CHECK(psr_noise < 6.0);
}

VTT_TEST(hann_window_tapers_to_zero) {
    const cv::Mat w = spectral::hann_window(cv::Size(16, 16));
    CHECK_NEAR(w.at<float>(0, 0), 0.0, 1e-6);
    CHECK(w.at<float>(8, 8) > 0.95F);
}

VTT_TEST_MAIN()
