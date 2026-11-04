/// @file spectral.hpp
/// @brief Frequency-domain helpers shared by the correlation filters.
#pragma once

#include <opencv2/core.hpp>

namespace vtt::spectral {

/// 2-D Hann (cosine) window, CV_32F.
[[nodiscard]] cv::Mat hann_window(cv::Size size);

/// Gaussian peak of standard deviation @p sigma centred at (width/2, height/2), CV_32F.
[[nodiscard]] cv::Mat gaussian_peak(cv::Size size, double sigma);

/// Forward DFT of a real CV_32F matrix, returning a full CV_32FC2 spectrum.
[[nodiscard]] cv::Mat fft(const cv::Mat& real);

/// Inverse DFT of a conjugate-symmetric CV_32FC2 spectrum, returning a scaled real CV_32F matrix.
[[nodiscard]] cv::Mat ifft_real(const cv::Mat& spectrum);

/// Element-wise complex product a * b (or a * conj(b) when @p conjugate_b is set).
[[nodiscard]] cv::Mat multiply(const cv::Mat& a, const cv::Mat& b, bool conjugate_b = false);

/// Element-wise complex division numerator / (denominator + regularization).
[[nodiscard]] cv::Mat divide(const cv::Mat& numerator, const cv::Mat& denominator,
                             double regularization = 0.0);

/// Sum over all elements of |z|^2 for a CV_32FC2 spectrum.
[[nodiscard]] double energy(const cv::Mat& spectrum);

/// Correlation peak with sub-pixel refinement.
struct Peak {
    cv::Point integer_location;  ///< Arg-max of the response.
    cv::Point2d location;        ///< Sub-pixel location from a parabolic fit.
    double value = 0.0;          ///< Response value at the arg-max.
};

/// Locate the maximum of a CV_32F response map with parabolic sub-pixel refinement.
[[nodiscard]] Peak find_peak(const cv::Mat& response);

/// Peak-to-sidelobe ratio: (peak - mean_sidelobe) / std_sidelobe, excluding a
/// (2 * exclusion_radius + 1)^2 window around the peak.
[[nodiscard]] double peak_to_sidelobe_ratio(const cv::Mat& response, const Peak& peak,
                                            int exclusion_radius = 5);

}  // namespace vtt::spectral
