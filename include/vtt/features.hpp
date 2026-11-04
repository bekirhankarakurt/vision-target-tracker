/// @file features.hpp
/// @brief Feature extraction for the correlation filters (grayscale and gradients, no HOG).
#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace vtt::features {

/// MOSSE preprocessing: log transform, zero mean / unit variance, cosine window.
/// @param patch CV_32F grayscale patch in [0, 1].
/// @param window Hann window of the same size.
[[nodiscard]] cv::Mat mosse_preprocess(const cv::Mat& patch, const cv::Mat& window);

/// KCF feature channels: zero-mean intensity plus horizontal and vertical
/// Sobel gradients, each multiplied by the cosine window.
[[nodiscard]] std::vector<cv::Mat> kcf_channels(const cv::Mat& patch, const cv::Mat& window);

}  // namespace vtt::features
