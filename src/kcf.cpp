#include "vtt/kcf.hpp"

#include "vtt/features.hpp"
#include "vtt/spectral.hpp"

#include <opencv2/imgproc.hpp>

namespace vtt {

KcfFilter::KcfFilter(const FilterParams& params) : params_(params) {}

KcfFilter::Spectra KcfFilter::extract(const cv::Mat& patch) const {
    Spectra spectra;
    for (const cv::Mat& channel : features::kcf_channels(patch, window_)) {
        spectra.push_back(spectral::fft(channel));
    }
    return spectra;
}

cv::Mat KcfFilter::kernel_correlation(const Spectra& xf, const Spectra& zf) const {
    CV_Assert(xf.size() == zf.size() && !xf.empty());
    const double n = static_cast<double>(size_.area());
    double xx = 0.0;
    double zz = 0.0;
    cv::Mat cross = cv::Mat::zeros(size_, CV_32FC2);
    for (std::size_t c = 0; c < xf.size(); ++c) {
        xx += spectral::energy(xf[c]);
        zz += spectral::energy(zf[c]);
        cross += spectral::multiply(zf[c], xf[c], true);
    }
    // Parseval: ||x||^2 = sum |x^|^2 / N.
    xx /= n;
    zz /= n;
    const cv::Mat xz = spectral::ifft_real(cross);
    const double denominator =
        params_.kernel_sigma * params_.kernel_sigma * n * static_cast<double>(xf.size());
    cv::Mat distance = (xx + zz) - 2.0 * xz;
    cv::max(distance, 0.0, distance);
    cv::Mat kernel;
    cv::exp(distance * (-1.0 / denominator), kernel);
    return spectral::fft(kernel);
}

cv::Mat KcfFilter::train(const Spectra& xf) const {
    return spectral::divide(target_spectrum_, kernel_correlation(xf, xf), params_.regularization);
}

void KcfFilter::init(const cv::Mat& patch) {
    CV_Assert(patch.type() == CV_32F && patch.cols >= 4 && patch.rows >= 4);
    size_ = patch.size();
    window_ = spectral::hann_window(size_);
    target_spectrum_ = spectral::fft(spectral::gaussian_peak(size_, params_.output_sigma));
    model_xf_ = extract(patch);
    model_alphaf_ = train(model_xf_);
}

cv::Mat KcfFilter::respond(const cv::Mat& patch) const {
    CV_Assert(!model_alphaf_.empty() && patch.size() == size_);
    const cv::Mat kzf = kernel_correlation(model_xf_, extract(patch));
    return spectral::ifft_real(spectral::multiply(model_alphaf_, kzf));
}

void KcfFilter::update(const cv::Mat& patch) {
    CV_Assert(!model_alphaf_.empty() && patch.size() == size_);
    const Spectra xf = extract(patch);
    const cv::Mat alphaf = train(xf);
    const double eta = params_.learning_rate;
    for (std::size_t c = 0; c < xf.size(); ++c) {
        model_xf_[c] = (1.0 - eta) * model_xf_[c] + eta * xf[c];
    }
    model_alphaf_ = (1.0 - eta) * model_alphaf_ + eta * alphaf;
}

}  // namespace vtt
