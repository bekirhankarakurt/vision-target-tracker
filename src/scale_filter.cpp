#include "vtt/scale_filter.hpp"

#include "vtt/spectral.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vtt {

namespace {

/// Crop @p size around @p center with replicated borders and resize to @p out_size.
cv::Mat sample_patch(const cv::Mat& gray, const cv::Point2d& center, const cv::Size2d& size,
                     cv::Size out_size) {
    const int w = std::max(2, cvRound(size.width));
    const int h = std::max(2, cvRound(size.height));
    const cv::Rect roi(cvRound(center.x - 0.5 * size.width), cvRound(center.y - 0.5 * size.height), w,
                       h);
    const cv::Rect inside = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    cv::Mat crop;
    if (inside.area() == 0) {
        crop = cv::Mat(h, w, CV_8U, cv::Scalar(cv::mean(gray)[0]));
    } else if (inside == roi) {
        crop = gray(roi);
    } else {
        cv::copyMakeBorder(gray(inside), crop, inside.y - roi.y, roi.br().y - inside.br().y,
                           inside.x - roi.x, roi.br().x - inside.br().x, cv::BORDER_REPLICATE);
    }
    cv::Mat resized;
    cv::resize(crop, resized, out_size, 0.0, 0.0,
               w > out_size.width ? cv::INTER_AREA : cv::INTER_LINEAR);
    cv::Mat out;
    resized.convertTo(out, CV_32F, 1.0 / 255.0);
    return out;
}

}  // namespace

ScaleFilter::ScaleFilter(const ScaleFilterParams& params) : params_(params) {
    if (params_.num_scales < 3 || params_.num_scales % 2 == 0 || params_.scale_step <= 1.0) {
        throw std::invalid_argument("scale filter needs an odd number (>= 3) of levels and step > 1");
    }
    const int half = params_.num_scales / 2;
    for (int i = -half; i <= half; ++i) {
        factors_.push_back(std::pow(params_.scale_step, static_cast<double>(i)));
    }
    const cv::Size size(params_.num_scales, 1);
    cv::Mat hann(size, CV_32F);
    cv::Mat label(size, CV_32F);
    const double sigma = params_.output_sigma * std::sqrt(static_cast<double>(params_.num_scales));
    for (int i = 0; i < params_.num_scales; ++i) {
        const auto x = static_cast<double>(i);
        const auto n = static_cast<double>(params_.num_scales);
        hann.at<float>(0, i) = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * CV_PI * (x + 1.0) / (n + 1.0)));
        const auto d = static_cast<double>(i - half);
        label.at<float>(0, i) = static_cast<float>(std::exp(-0.5 * d * d / (sigma * sigma)));
    }
    scale_window_ = hann;
    label_spectrum_ = spectral::fft(label);
}

cv::Mat ScaleFilter::samples(const cv::Mat& gray, const cv::Point2d& center,
                             const cv::Size2d& target_size) const {
    const int count = params_.num_scales;
    std::vector<cv::Mat> rows;
    rows.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
        const double f = factors_[static_cast<std::size_t>(k)];
        const cv::Mat patch = sample_patch(gray, center,
                                           cv::Size2d(target_size.width * f, target_size.height * f),
                                           model_size_);
        cv::Mat gx;
        cv::Mat gy;
        cv::Sobel(patch, gx, CV_32F, 1, 0, 3, 0.25, 0.0, cv::BORDER_REPLICATE);
        cv::Sobel(patch, gy, CV_32F, 0, 1, 3, 0.25, 0.0, cv::BORDER_REPLICATE);
        const cv::Mat intensity = patch - cv::mean(patch)[0];
        const std::vector<cv::Mat> parts{intensity.reshape(1, 1), gx.reshape(1, 1), gy.reshape(1, 1)};
        cv::Mat row;
        cv::hconcat(parts, row);
        rows.emplace_back(row * static_cast<double>(scale_window_.at<float>(0, k)));
    }
    cv::Mat stacked;
    cv::vconcat(rows, stacked);
    return stacked.t();
}

void ScaleFilter::train(const cv::Mat& spectrum, double rate) {
    cv::Mat labels;
    cv::repeat(label_spectrum_, spectrum.rows, 1, labels);
    const cv::Mat numerator = spectral::multiply(labels, spectrum, true);
    const cv::Mat power = spectral::multiply(spectrum, spectrum, true);
    cv::Mat denominator;
    cv::reduce(power, denominator, 0, cv::REDUCE_SUM, CV_32FC2);
    if (rate >= 1.0) {
        numerator_ = numerator;
        denominator_ = denominator;
    } else {
        numerator_ = (1.0 - rate) * numerator_ + rate * numerator;
        denominator_ = (1.0 - rate) * denominator_ + rate * denominator;
    }
}

void ScaleFilter::init(const cv::Mat& gray, const cv::Point2d& center,
                       const cv::Size2d& target_size) {
    CV_Assert(gray.type() == CV_8U && target_size.width > 0.0 && target_size.height > 0.0);
    const double s = std::sqrt(static_cast<double>(params_.model_area) /
                               (target_size.width * target_size.height));
    model_size_ = cv::Size(std::max(4, cvRound(target_size.width * s)),
                           std::max(4, cvRound(target_size.height * s)));
    cv::Mat spectrum;
    cv::dft(samples(gray, center, target_size), spectrum, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    train(spectrum, 1.0);
}

double ScaleFilter::estimate(const cv::Mat& gray, const cv::Point2d& center,
                             const cv::Size2d& target_size) const {
    CV_Assert(initialized());
    cv::Mat spectrum;
    cv::dft(samples(gray, center, target_size), spectrum, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    cv::Mat summed;
    cv::reduce(spectral::multiply(spectrum, numerator_), summed, 0, cv::REDUCE_SUM, CV_32FC2);
    const double mean_energy = cv::mean(denominator_)[0];
    const cv::Mat response =
        spectral::ifft_real(spectral::divide(summed, denominator_, params_.regularization * mean_energy));
    const spectral::Peak peak = spectral::find_peak(response);
    const int half = params_.num_scales / 2;
    const auto limit = static_cast<double>(half);
    const double index = std::clamp(peak.location.x - limit, -limit, limit);
    return std::pow(params_.scale_step, index);
}

void ScaleFilter::update(const cv::Mat& gray, const cv::Point2d& center,
                         const cv::Size2d& target_size) {
    CV_Assert(initialized());
    cv::Mat spectrum;
    cv::dft(samples(gray, center, target_size), spectrum, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    train(spectrum, params_.learning_rate);
}

}  // namespace vtt
