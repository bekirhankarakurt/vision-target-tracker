/// @file scale_filter.hpp
/// @brief One-dimensional correlation filter over a scale pyramid (DSST-style
///        discriminative scale estimation, Danelljan et al., BMVC 2014).
#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace vtt {

struct ScaleFilterParams {
    int num_scales = 15;          ///< Pyramid levels (odd).
    double scale_step = 1.025;    ///< Ratio between consecutive levels.
    double output_sigma = 0.25;   ///< Gaussian label std-dev, in units of sqrt(num_scales).
    double learning_rate = 0.025;
    double regularization = 1e-2;
    int model_area = 512;         ///< Pixel area each sample is resized to.
};

/// Learns how the target's appearance changes across a pyramid of scales and
/// returns the relative scale change that maximises the 1-D correlation response.
class ScaleFilter {
public:
    explicit ScaleFilter(const ScaleFilterParams& params = {});

    /// Train on the target of size @p target_size centred at @p center in an 8-bit gray frame.
    void init(const cv::Mat& gray, const cv::Point2d& center, const cv::Size2d& target_size);
    /// Relative scale factor (1.0 = unchanged) for the target currently of @p target_size.
    [[nodiscard]] double estimate(const cv::Mat& gray, const cv::Point2d& center,
                                  const cv::Size2d& target_size) const;
    void update(const cv::Mat& gray, const cv::Point2d& center, const cv::Size2d& target_size);

    [[nodiscard]] bool initialized() const noexcept { return !denominator_.empty(); }
    [[nodiscard]] const std::vector<double>& factors() const noexcept { return factors_; }

private:
    /// D x S matrix: one feature column per scale level, Hann-weighted across scales.
    [[nodiscard]] cv::Mat samples(const cv::Mat& gray, const cv::Point2d& center,
                                  const cv::Size2d& target_size) const;
    void train(const cv::Mat& spectrum, double rate);

    ScaleFilterParams params_;
    std::vector<double> factors_;
    cv::Mat scale_window_;  ///< 1 x S
    cv::Mat label_spectrum_;  ///< 1 x S complex
    cv::Size model_size_;
    cv::Mat numerator_;     ///< D x S complex
    cv::Mat denominator_;   ///< 1 x S complex (real valued)
};

}  // namespace vtt
