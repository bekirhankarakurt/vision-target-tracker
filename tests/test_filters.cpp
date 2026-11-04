#include "test_harness.hpp"

#include "vtt/correlation_filter.hpp"
#include "vtt/kcf.hpp"
#include "vtt/mosse.hpp"
#include "vtt/spectral.hpp"

#include <opencv2/imgproc.hpp>

#include <array>

using namespace vtt;

namespace {

cv::Mat textured_scene(int seed) {
    cv::Mat noise(200, 200, CV_32F);
    cv::RNG rng(static_cast<std::uint64_t>(seed));
    rng.fill(noise, cv::RNG::UNIFORM, 0.0, 1.0);
    cv::GaussianBlur(noise, noise, cv::Size(), 2.0);
    cv::normalize(noise, noise, 0.0, 1.0, cv::NORM_MINMAX);
    return noise;
}

cv::Mat crop(const cv::Mat& scene, cv::Point2d center, cv::Size size) {
    cv::Mat patch;
    cv::getRectSubPix(scene, size, cv::Point2f(static_cast<float>(center.x), static_cast<float>(center.y)),
                      patch);
    return patch;
}

/// Train at the centre, then evaluate on a window displaced by -shift: the target
/// appears at +shift and the peak must move accordingly.
void check_translation(Algorithm algorithm) {
    const cv::Mat scene = textured_scene(1);
    const cv::Size size(64, 64);
    const cv::Point2d center(100.0, 100.0);
    auto filter = make_filter(algorithm, default_filter_params(algorithm));
    filter->init(crop(scene, center, size));
    CHECK(filter->patch_size() == size);
    CHECK(filter->algorithm() == algorithm);

    const std::array<cv::Point2d, 4> shifts{{{0.0, 0.0}, {5.0, -3.0}, {-7.5, 4.25}, {2.0, 9.0}}};
    for (const cv::Point2d& shift : shifts) {
        const cv::Mat response = filter->respond(crop(scene, center - shift, size));
        const spectral::Peak peak = spectral::find_peak(response);
        CHECK_NEAR(peak.location.x - 32.0, shift.x, 0.6);
        CHECK_NEAR(peak.location.y - 32.0, shift.y, 0.6);
        CHECK(spectral::peak_to_sidelobe_ratio(response, peak) > 8.0);
    }
    const cv::Mat other = crop(textured_scene(99), center, size);
    const cv::Mat response = filter->respond(other);
    CHECK(spectral::peak_to_sidelobe_ratio(response, spectral::find_peak(response)) < 8.0);
}

}  // namespace

VTT_TEST(mosse_locates_translated_target) { check_translation(Algorithm::Mosse); }

VTT_TEST(kcf_locates_translated_target) { check_translation(Algorithm::Kcf); }

VTT_TEST(update_adapts_to_appearance_change) {
    const cv::Mat scene = textured_scene(2);
    const cv::Size size(48, 40);
    const cv::Point2d center(100.0, 100.0);
    for (const Algorithm algorithm : {Algorithm::Mosse, Algorithm::Kcf}) {
        auto filter = make_filter(algorithm, default_filter_params(algorithm));
        filter->init(crop(scene, center, size));
        // Gradual contrast reduction and brightening of the target.
        cv::Mat faded;
        for (int i = 1; i <= 20; ++i) {
            const double gain = 1.0 - 0.02 * i;
            faded = crop(scene, center, size) * gain + 0.01 * i;
            filter->update(faded);
        }
        const cv::Mat shifted = crop(scene, center - cv::Point2d(3.0, 2.0), size) * 0.6 + 0.2;
        const cv::Mat response = filter->respond(shifted);
        const spectral::Peak peak = spectral::find_peak(response);
        CHECK_NEAR(peak.location.x - 24.0, 3.0, 0.75);
        CHECK_NEAR(peak.location.y - 20.0, 2.0, 0.75);
        CHECK(spectral::peak_to_sidelobe_ratio(response, peak) > 8.0);
    }
}

VTT_TEST(default_params_differ_per_algorithm) {
    const FilterParams mosse = default_filter_params(Algorithm::Mosse);
    const FilterParams kcf = default_filter_params(Algorithm::Kcf);
    CHECK(mosse.learning_rate > 0.0 && kcf.learning_rate > 0.0);
    CHECK(kcf.regularization < mosse.regularization);
}

VTT_TEST_MAIN()
