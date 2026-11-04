/// @file kcf.hpp
/// @brief Kernelized Correlation Filter with a Gaussian kernel (Henriques et al., TPAMI 2015).
#pragma once

#include "vtt/correlation_filter.hpp"

#include <vector>

namespace vtt {

/// Multi-channel KCF on intensity + gradient features with a Gaussian kernel.
class KcfFilter final : public CorrelationFilter {
public:
    explicit KcfFilter(const FilterParams& params);

    void init(const cv::Mat& patch) override;
    [[nodiscard]] cv::Mat respond(const cv::Mat& patch) const override;
    void update(const cv::Mat& patch) override;

    [[nodiscard]] Algorithm algorithm() const noexcept override { return Algorithm::Kcf; }
    [[nodiscard]] cv::Size patch_size() const noexcept override { return size_; }

private:
    using Spectra = std::vector<cv::Mat>;

    [[nodiscard]] Spectra extract(const cv::Mat& patch) const;
    /// Spectrum of the Gaussian kernel correlation k^{xz}.
    [[nodiscard]] cv::Mat kernel_correlation(const Spectra& xf, const Spectra& zf) const;
    [[nodiscard]] cv::Mat train(const Spectra& xf) const;

    FilterParams params_;
    cv::Size size_;
    cv::Mat window_;
    cv::Mat target_spectrum_;  ///< y^
    Spectra model_xf_;         ///< x^ (per channel)
    cv::Mat model_alphaf_;     ///< alpha^
};

}  // namespace vtt
