#include "vtt/spectral.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace vtt::spectral {

cv::Mat hann_window(cv::Size size) {
    cv::Mat window;
    cv::createHanningWindow(window, size, CV_32F);
    return window;
}

cv::Point2d response_center(cv::Size size) noexcept {
    const int cx = size.width / 2;
    const int cy = size.height / 2;
    return {static_cast<double>(cx), static_cast<double>(cy)};
}

cv::Mat gaussian_peak(cv::Size size, double sigma) {
    cv::Mat peak(size, CV_32F);
    const cv::Point2d c = response_center(size);
    const double cx = c.x;
    const double cy = c.y;
    const double scale = -0.5 / (sigma * sigma);
    for (int y = 0; y < size.height; ++y) {
        auto* row = peak.ptr<float>(y);
        const double dy = static_cast<double>(y) - cy;
        for (int x = 0; x < size.width; ++x) {
            const double dx = static_cast<double>(x) - cx;
            row[x] = static_cast<float>(std::exp((dx * dx + dy * dy) * scale));
        }
    }
    return peak;
}

cv::Mat fft(const cv::Mat& real) {
    cv::Mat spectrum;
    cv::dft(real, spectrum, cv::DFT_COMPLEX_OUTPUT);
    return spectrum;
}

cv::Mat ifft_real(const cv::Mat& spectrum) {
    cv::Mat real;
    cv::dft(spectrum, real, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return real;
}

cv::Mat multiply(const cv::Mat& a, const cv::Mat& b, bool conjugate_b) {
    cv::Mat product;
    cv::mulSpectrums(a, b, product, 0, conjugate_b);
    return product;
}

cv::Mat divide(const cv::Mat& numerator, const cv::Mat& denominator, double regularization) {
    CV_Assert(numerator.type() == CV_32FC2 && denominator.type() == CV_32FC2);
    CV_Assert(numerator.size() == denominator.size());
    cv::Mat quotient(numerator.size(), CV_32FC2);
    const auto reg = static_cast<float>(regularization);
    for (int y = 0; y < numerator.rows; ++y) {
        const auto* n = numerator.ptr<cv::Vec2f>(y);
        const auto* d = denominator.ptr<cv::Vec2f>(y);
        auto* q = quotient.ptr<cv::Vec2f>(y);
        for (int x = 0; x < numerator.cols; ++x) {
            const float dr = d[x][0] + reg;
            const float di = d[x][1];
            const float inv = 1.0F / std::max(dr * dr + di * di, 1e-12F);
            q[x][0] = (n[x][0] * dr + n[x][1] * di) * inv;
            q[x][1] = (n[x][1] * dr - n[x][0] * di) * inv;
        }
    }
    return quotient;
}

double energy(const cv::Mat& spectrum) {
    CV_Assert(spectrum.type() == CV_32FC2);
    return cv::norm(spectrum, cv::NORM_L2SQR);
}

namespace {

double parabolic_offset(double left, double center, double right) {
    const double denominator = left - 2.0 * center + right;
    if (std::abs(denominator) < 1e-12) {
        return 0.0;
    }
    return std::clamp(0.5 * (left - right) / denominator, -0.5, 0.5);
}

}  // namespace

Peak find_peak(const cv::Mat& response) {
    CV_Assert(response.type() == CV_32F && !response.empty());
    Peak peak;
    cv::minMaxLoc(response, nullptr, &peak.value, nullptr, &peak.integer_location);
    const int x = peak.integer_location.x;
    const int y = peak.integer_location.y;
    const int w = response.cols;
    const int h = response.rows;
    peak.location = cv::Point2d(x, y);
    if (w >= 3) {
        const auto left = static_cast<double>(response.at<float>(y, (x - 1 + w) % w));
        const auto right = static_cast<double>(response.at<float>(y, (x + 1) % w));
        peak.location.x += parabolic_offset(left, peak.value, right);
    }
    if (h >= 3) {
        const auto up = static_cast<double>(response.at<float>((y - 1 + h) % h, x));
        const auto down = static_cast<double>(response.at<float>((y + 1) % h, x));
        peak.location.y += parabolic_offset(up, peak.value, down);
    }
    return peak;
}

double peak_to_sidelobe_ratio(const cv::Mat& response, const Peak& peak, int exclusion_radius) {
    CV_Assert(response.type() == CV_32F);
    double sum = 0.0;
    double sum_sq = 0.0;
    long count = 0;
    const cv::Point p = peak.integer_location;
    for (int y = 0; y < response.rows; ++y) {
        const auto* row = response.ptr<float>(y);
        const bool row_excluded = std::abs(y - p.y) <= exclusion_radius;
        for (int x = 0; x < response.cols; ++x) {
            if (row_excluded && std::abs(x - p.x) <= exclusion_radius) {
                continue;
            }
            const auto v = static_cast<double>(row[x]);
            sum += v;
            sum_sq += v * v;
            ++count;
        }
    }
    if (count < 2) {
        return 0.0;
    }
    const auto n = static_cast<double>(count);
    const double mean = sum / n;
    const double variance = std::max(sum_sq / n - mean * mean, 0.0);
    return (peak.value - mean) / (std::sqrt(variance) + 1e-9);
}

}  // namespace vtt::spectral
