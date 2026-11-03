#include "vtt/mosse.hpp"

#include "vtt/features.hpp"
#include "vtt/spectral.hpp"

#include <opencv2/imgproc.hpp>

namespace vtt {

namespace {

constexpr int kAugmentedSamples = 8;
constexpr double kMaxRotationRad = 0.15;
constexpr double kMaxScaleJitter = 0.05;

/// Small random rotation/scale about the patch centre (MOSSE initialisation).
cv::Mat perturb(const cv::Mat& patch, cv::RNG& rng) {
    const cv::Point2f center(static_cast<float>(patch.cols) * 0.5F,
                             static_cast<float>(patch.rows) * 0.5F);
    const double angle_deg = rng.uniform(-kMaxRotationRad, kMaxRotationRad) * 180.0 / CV_PI;
    const double scale = 1.0 + rng.uniform(-kMaxScaleJitter, kMaxScaleJitter);
    const cv::Mat warp = cv::getRotationMatrix2D(center, angle_deg, scale);
    cv::Mat out;
    cv::warpAffine(patch, out, warp, patch.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    return out;
}

}  // namespace

MosseFilter::MosseFilter(const FilterParams& params) : params_(params) {}

void MosseFilter::init(const cv::Mat& patch) {
    CV_Assert(patch.type() == CV_32F && patch.cols >= 4 && patch.rows >= 4);
    size_ = patch.size();
    window_ = spectral::hann_window(size_);
    target_spectrum_ = spectral::fft(spectral::gaussian_peak(size_, params_.output_sigma));

    const cv::Mat f = spectral::fft(features::mosse_preprocess(patch, window_));
    numerator_ = spectral::multiply(target_spectrum_, f, true);
    denominator_ = spectral::multiply(f, f, true);

    cv::RNG rng(0x5EED);
    for (int i = 0; i < kAugmentedSamples; ++i) {
        const cv::Mat fi = spectral::fft(features::mosse_preprocess(perturb(patch, rng), window_));
        numerator_ += spectral::multiply(target_spectrum_, fi, true);
        denominator_ += spectral::multiply(fi, fi, true);
    }
    refresh_filter();
}

cv::Mat MosseFilter::respond(const cv::Mat& patch) const {
    CV_Assert(!filter_.empty() && patch.size() == size_);
    const cv::Mat f = spectral::fft(features::mosse_preprocess(patch, window_));
    return spectral::ifft_real(spectral::multiply(f, filter_));
}

void MosseFilter::update(const cv::Mat& patch) {
    CV_Assert(!filter_.empty() && patch.size() == size_);
    const cv::Mat f = spectral::fft(features::mosse_preprocess(patch, window_));
    const double eta = params_.learning_rate;
    numerator_ = (1.0 - eta) * numerator_ + eta * spectral::multiply(target_spectrum_, f, true);
    denominator_ = (1.0 - eta) * denominator_ + eta * spectral::multiply(f, f, true);
    refresh_filter();
}

void MosseFilter::refresh_filter() {
    const double mean_energy = cv::mean(denominator_)[0];
    filter_ = spectral::divide(numerator_, denominator_, params_.regularization * mean_energy);
}

}  // namespace vtt
